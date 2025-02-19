import io, sys, os, time, random, logging
import asyncio
from hashlib import md5
from typing import Union, List, Tuple
sys.path.append('/home/noah/projects/cvInfra/src')

import cv2
import numpy as np
import gradio as gr
from faker import Faker
from user_agents import parse

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

os.environ["COS_SECRET_ID"] = ""
os.environ["COS_SECRET_KEY"] = ""

os.environ["WEB_DEMO"] = "True" # local testing

secret_id = os.environ['COS_SECRET_ID']
secret_key = os.environ['COS_SECRET_KEY']


from app.stitching_v2.lib.stitch import main_stitching, ml_stitch_im_video
from app.stitching_v2.lib.nn import ExtrMatcher, OrtMatcher, OrtFeatureExtractor


class NN_Model:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_kpts = 2048
        resolution = "1280x720"
        # resolution = "960x540"
        # resolution = "640x360"
        self.extractor = OrtFeatureExtractor(
            providers=['CPUExecutionProvider'],
            model_path=f"/home/noah/projects/cvInfra/src/app/stitching_v2/lib/weights/onnx/xfeat_{n_kpts}_{resolution}.onnx")
        self.matcher = OrtMatcher(
            providers=['CPUExecutionProvider'],
            score_threshold=0.7,
            model_path=f"/home/noah/projects/cvInfra/src/app/stitching_v2/lib/weights/onnx/lighterglue_L3.onnx")


class S3_Client:
    def __init__(self, *args, **kwargs):
        config = CosConfig(Region='ap-shanghai', SecretId=secret_id, SecretKey=secret_key, Token=None, Scheme='https')
        self.client = CosS3Client(config)
        self.region = "ap-shanghai"
        self.bucket = "retail-dev-cn-1304042378"
        self.video_prefix = "stitchv2/videos"
        self.pano_prefix = "stitchv2/panos"
        # download path prefix
        self.end_point = f"https://{self.bucket}.cos.{self.region}.myqcloud.com"
        # 404 image
        self.no_image = f"{self.end_point}/stitchv2/no_image.jpg"

    def get_url(self, user_name: str, md5_key: str):
        key = f"{self.pano_prefix}/{self.user_name(user_name)}/{self.get_datetime()}/{md5_key}.png"
        if self._is_exit(key):
            return f"{self.end_point}/{key}"
        else:
            return ""

    def upload_npimg(self, m_img: np.ndarray, user_name: str, md5_key: str):
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGRA2RGB)
        to_upload_img = io.BytesIO(cv2.imencode('.png', m_img)[1]).getvalue()

        key = f"{self.pano_prefix}/{self.user_name(user_name)}/{self.get_datetime()}/{md5_key}.png"
        if self._is_exit(key):
            logging.info(f"File {key} already exists.")
            return

        response = self.client.put_object(
            Bucket=self.bucket,
            Body=to_upload_img,
            Key=key,
        )
        if 'ETag' in response:
            logging.info(f"Image {key} uploaded successfully.")
            return response['ETag']
        else:
            logging.error(f"Failed to upload image {key}")

        return response

    def upload_file(self, file_path: str, user_name: str, md5_key: str=None):
        if md5_key is None:
            md5_key = self.get_md5(file_path)

        filename, ext = os.path.splitext(os.path.basename(file_path))
        md5_path = f"{md5_key}{ext}"

        key = f"{self.video_prefix}/{self.user_name(user_name)}/{self.get_datetime()}/{md5_path}"
        if self._is_exit(key):
            logging.info(f"File {key} already exists.")
            return

        response = self.client.upload_file(
            Bucket=self.bucket,
            LocalFilePath=file_path,
            Key=key,
            PartSize=1,
            MAXThread=10,
            EnableMD5=False, # skip md5 check
        )
        if 'ETag' in response:
            logging.info(f"Video {key} uploaded successfully.")
            # gr.Info(f"Video {filename} uploaded successfully.")
            return response['ETag']
        else:
            # gr.Error(f"Failed to upload video {filename}")
            logging.error(f"Failed to upload video {key}")

        return response

    def _is_exit(self, key):
        response = self.client.object_exists(
            Bucket=self.bucket,
            Key=key,
        )
        return response

    def list_objects(self, user_name: str):
        # filter space and special characters, keep only letters and numbers
        user_name = self.user_name(user_name)
        response = self.client.list_objects(
            Bucket=self.bucket,
            Prefix=f'{self.pano_prefix}/{user_name}/',
        )
        # get image list
        im_dict = {}
        img_list = []
        if 'Contents' in response:
            for content in response['Contents']:
                if content['Key'].endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    last_modified = content['LastModified'] # 2025-02-19T09:19:14.000Z
                    # convert to timestamp
                    last_modified = time.mktime(time.strptime(last_modified, "%Y-%m-%dT%H:%M:%S.000Z"))
                    im_dict[content['Key']] = last_modified

            # sort by last modified
            sorted_im_dict = dict(sorted(im_dict.items(), key=lambda item: item[1], reverse=True))
            for key in sorted_im_dict.keys():
                im_link = "{}/{}".format(self.end_point, key)
                print(im_link)
                img_list.append(im_link)

        if len(img_list) == 0:
            logging.info(f"No images found. User: {user_name}")
            logging.info(f"no image: {self.no_image}")
            img_list.append(self.no_image)

        return img_list

    @staticmethod
    def user_name(user_name: str):
        return ''.join(e for e in user_name if e.isalnum())

    @staticmethod
    def get_md5(fname: str) -> str:
        with open(fname, "rb") as f:
            content = f.read()
            md5_hash = md5(content).hexdigest()
        return md5_hash

    @staticmethod
    def get_user_info(request: gr.Request):
        user_agent = request.headers.get("User-Agent", "")
        user_agent_info = parse(user_agent)
        info = {
            "browser": {
                "family": user_agent_info.browser.family,
                "version": user_agent_info.browser.version_string
            },
            "device": user_agent_info.device.family,
            "os": {
                "family": user_agent_info.os.family,
                "version": user_agent_info.os.version_string
            }
        }
        return str(info)

    @staticmethod
    def get_datetime():
        return time.strftime("%Y%m%d", time.localtime())


def get_random_name():
    faker = Faker(locale=['it_IT', 'en_US'])
    _name = faker.name().split()[0].lower()
    return ''.join(e for e in _name if e.isalnum())


def info_search(user_name: str):
    try:
        gr.Info("正在查询中...请勿重复点击")
        s3_client = S3_Client()
        res = s3_client.list_objects(user_name)
        return res
    except Exception as e:
        logging.error(f"Failed to search images: {str(e)}")
        gr.Error("查询失败，请稍后重试")
        return []


def process_video(files: list, user_name: str, progress=gr.Progress()):
    results = []
    info_text = "没有视频文件"

    if not files:
        return results, info_text

    s3client = S3_Client()
    nn_model = NN_Model()

    for idx, file_path in enumerate(files):
        info_text = f"开始处理视频 {idx + 1}/{len(files)}"
        progress(idx / len(files), desc=info_text)
        pano = None

        try:
            # is pano already exists, return url
            url = s3client.get_url(user_name, s3client.get_md5(file_path))
            if url != "":
                logging.info(f"File {file_path} already exists.")
                pano = url
            else:
                # stitch video
                pano = ml_stitch_im_video(file_path, nn_model)
                s3client.upload_npimg(pano, user_name, s3client.get_md5(file_path))

        except Exception as e:
            logging.error(f"Failed to process video {file_path}: {str(e)}")
            gr.Error(f"视频 {file_path} 处理失败: {str(e)}")

        results.append(pano)
        info_text = f"视频 {idx + 1}/{len(files)} 处理完成"
        progress((idx + 1) / len(files), desc=info_text)
        yield results, info_text
    
    info_text = f"共计 {len(files)} 个视频处理完成"
    progress(1, desc=info_text)
    gr.Info(info_text)
    yield results, "拼图完成！"
    # return results, info_text


def upload_with_progress(files: list, user_name: str, progress=gr.Progress()):
    """处理文件上传并实时更新进度的异步生成器"""
    if not files:
        yield [], "没有视频文件"

    s3client = S3_Client()

    results = []  # 存储处理后的图像结果
    total = len(files)
    completed = 0
    info_update = "等待上传..."
    progress(0.05, desc="开始上传...")

    for idx, file_path in enumerate(files):
        info_text = f"开始上传视频 {idx + 1}/{len(files)}"
        progress(idx / len(files), desc=info_text)

        s3client.upload_file(file_path, user_name)
        completed += 1
        info_update = f"上传进度 {completed}/{total}"
        progress(completed / total, desc=info_update)
        yield results, info_update

    info_update = f"上传完成！共上传 {total} 个文件"
    progress(1, desc=info_update)
    gr.Info(info_update)

    return results, info_update


####### INTERFACE #######
with gr.Blocks(
    title="Stitch v2 Demo",
) as demo:
    gr.Markdown("# 拼图数据采集")
    gr.Markdown("开始采集之前，请阅读 [数据采集指南](https://clobotics.atlassian.net/wiki/pages/resumedraft.action?draftId=1199439906)")
    gr.Markdown("点击下面的按钮上传视频，支持 `.mp4, .mov` 视频格式。数据上传成功后，系统会自动进行拼图。")
    gr.Markdown("拼图需要一些时间，无需拼图结果待视频上传成功后关闭页面，之后根据用户名查看。")
    # gr.Markdown(f"你的用户名: `{get_random_name()}`, 请妥善保存用户名以便查看拼图结果。")

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            user_name = gr.Textbox(label="用户名", value=get_random_name())
            btn_search = gr.Button(value="查看结果")
            file_input = gr.File(label="选择文件", file_count="multiple")

        with gr.Column(scale=2):
            upload_progress = gr.Textbox(label="上传进度", value="等待上传...")
            gallery = gr.Gallery(label="上传结果", preview=True)

    file_input.upload(
        fn=upload_with_progress,
        inputs=[file_input, user_name],
        outputs=[gallery, upload_progress],
    ).then(
        fn=process_video,
        inputs=[file_input, user_name],
        outputs=[gallery, upload_progress],
    )
    # search button
    btn_search.click(fn=info_search, inputs=user_name, outputs=gallery)


if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True, server_name="0.0.0.0", server_port=8081)
