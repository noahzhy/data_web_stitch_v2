import io, sys, os, time, random, logging, json
from dataclasses import dataclass
from datetime import datetime
import asyncio
from hashlib import md5
from typing import Union, List, Tuple, Dict, Any

import cv2
import numpy as np
import gradio as gr
from user_agents import parse

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client


from app.stitching_v2.lib.stitch import ml_stitch_im_video
from app.stitching_v2.lib.nn import ExtrMatcher, OrtMatcher, OrtFeatureExtractor


logging.basicConfig(level=logging.INFO, stream=sys.stdout)

secret_id = os.environ['COS_SECRET_ID']
secret_key = os.environ['COS_SECRET_KEY']


meta_data = {
    'version':              '2025.02.21',                           # update version
    'description':          'stitching video',                      # update description
    'group':                'retail',
    'task':                 'stitching',
    'platform':             'gradio',                               # collect data from gradio(web)
    #### USER
    'user_name':            '',
    'device':               '',
    'browser':              '',
    'os':                   '',
    #### DATETIME
    'datetime':             '',
    #### MD5
    'md5':                  '',
    #### VIDEO
    'video_url':            [],
    'video_res':            [],
    'video_duration':       [],
    #### STITCHING RESULT
    'image_url':            [],
    'image_res':            [],
    #### ML
    'extractor': {
        'model':            'onnx',
        'name':             'xfeat',
        'n_kpts':           2048,
        'resolution':       '1280x720',
    },
    'matcher': {
        'model':            'onnx',
        'name':             'lighterglue(L3)',
        'score_threshold':  0.7,
    },
    #### STORAGE
    'storage':              'qcloud_cos',
    'bucket':               'videostitch-demo-1304042378',
    'region':               'ap-shanghai',
    #### OTHER
    'memo':                 'demo test',
    'user_rate':            '',                                     # user rate, check if the image is useful
}

# dataclass MetaData, init via given dict
class MetaData:
    def __init__(self, data=meta_data):
        for key, value in data.items():
            setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def update(self, key, value=None):
        if isinstance(key, dict):
            for k, v in key.items():
                setattr(self, k, v)
        else:
            setattr(self, key, value)
        return self

    def get(self, key):
        return getattr(self, key, None)

    def to_json(self):
        logging.debug(f"Metadata: {self.__dict__}")
        return json.dumps(self.__dict__)


class NN_Model:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_kpts = 2048
        resolution = "1280x720"
        # resolution = "960x540"
        # resolution = "640x360"
        self.extractor = OrtFeatureExtractor(
            providers=['CPUExecutionProvider'],
            model_path=f"/src/app/stitching_v2/lib/weights/onnx/xfeat_{n_kpts}_{resolution}.onnx")
            # model_path=f"/home/noah/projects/cvInfra/src/app/stitching_v2/lib/weights/onnx/superpoint.onnx")
        self.matcher = OrtMatcher(
            providers=['CPUExecutionProvider'],
            score_threshold=0.7,
            model_path=f"/src/app/stitching_v2/lib/weights/onnx/lighterglue_L3.onnx")
            # model_path=f"/home/noah/projects/cvInfra/src/app/stitching_v2/lib/weights/onnx/superpoint_lightglue.onnx")


class S3_Client:
    def __init__(self, *args, **kwargs):
        config = CosConfig(Region='ap-shanghai', SecretId=secret_id, SecretKey=secret_key, Token=None, Scheme='https')
        self.client = CosS3Client(config)
        self.region = "ap-shanghai"
        self.bucket = "videostitch-demo-1304042378"
        self.video_prefix = "videos"
        self.image_prefix = "images"
        self.json_prefix = "jsons"
        self.meta_data = MetaData()
        # download path prefix
        self.end_point = f"https://{self.bucket}.cos.{self.region}.myqcloud.com"
        # 404 image
        self.no_image = f"{self.end_point}/no_image.jpg"

    def get_url(self, user_name: str, md5_key: str):
        key = f"{self.image_prefix}/{self.user_name(user_name)}/{self.get_datetime()}/{md5_key}.png"
        if self._is_exit(key):
            return f"{self.end_point}/{key}"
        else:
            return ""

    def _is_exit(self, key):
        response = self.client.object_exists(
            Bucket=self.bucket,
            Key=key,
        )
        return response

    def upload_metadata(self, user_name: str):
        meta_data = self.meta_data
        md5_key = meta_data.get('md5')
        if md5_key == '':
            logging.error("MD5 key is missing.")
            md5_key = "{}_{}".format(user_name, self.get_timestamp())

        meta_data.update({
            'md5': md5_key,
            'user_name': user_name,
            'datetime': self.get_timestamp(),
        })

        key = f"{self.json_prefix}/{md5_key}.json"
        response = self.client.put_object(
            Bucket=self.bucket,
            Body=meta_data.to_json(),
            Key=key,
        )
        if 'ETag' in response:
            logging.info(f"Metadata {key} uploaded successfully.")
            return response['ETag']
        else:
            logging.error(f"Failed to upload metadata {key}")

        return response

    def upload_npimg(self, m_img: np.ndarray, user_name: str, md5_key: str):
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGRA2RGB)
        to_upload_img = io.BytesIO(cv2.imencode('.png', m_img)[1]).getvalue()

        key = f"{self.image_prefix}/{self.user_name(user_name)}/{self.get_datetime()}/{md5_key}.png"
        if self._is_exit(key):
            logging.info(f"File {key} already exists.")
            return

        response = self.client.put_object(
            Bucket=self.bucket,
            Body=to_upload_img,
            Key=key,
        )
        im_url = ""
        if 'ETag' in response:
            logging.info(f"Image {key} uploaded successfully.")
            im_url = f"{self.end_point}/{key}"
        else:
            logging.error(f"Failed to upload image {key}")

        return im_url

    def upload_file(self, file_path: str, user_name: str, md5_key: str = None) -> str:
        """Upload a video file to S3 storage.
        
        Args:
            file_path: Path to the video file
            user_name: User name/email 
            md5_key: Optional MD5 hash key for the file
            
        Returns:
            str: URL of uploaded file if successful, empty string otherwise
        """
        # Validate inputs
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return ""

        try:
            # Generate key path
            md5_key = md5_key or self.get_md5(file_path)
            _, ext = os.path.splitext(file_path)
            key = f"{self.video_prefix}/{self.user_name(user_name)}/{self.get_datetime()}/{md5_key}{ext}"

            # Check if file already exists
            if self._is_exit(key):
                logging.info(f"File already exists: {key}")
                return f"{self.end_point}/{key}"

            logging.info(f"Uploading file: {file_path} to {key}")
            # Upload file
            response = self.client.upload_file(
                Bucket=self.bucket,
                LocalFilePath=file_path,
                Key=key,
                PartSize=1,
                MAXThread=10,
                EnableMD5=False,
            )

            if 'ETag' in response:
                url = f"{self.end_point}/{key}"
                logging.info(f"Upload successful: {url}")
                return url

            logging.error(f"Upload failed - no ETag in response")
            return ""

        except Exception as e:
            logging.exception(f"Upload failed: {str(e)}")
            return ""

    def list_objects(self, user_name: str) -> List[str]:
        # filter space and special characters, keep only letters and numbers
        user_name = self.user_name(user_name)
        logging.info("Check prefix: {}".format(f'{self.image_prefix}/{user_name}/'))
        response = self.client.list_objects(
            Bucket=self.bucket,
            Prefix=f'{self.image_prefix}/{user_name}/',
        )
        # get image list
        im_dict = {}
        img_list = []
        if 'Contents' in response:
            for content in response['Contents']:
                if content['Key'].endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    last_modified = content['LastModified'] # 2025-02-19T09:19:14.000Z
                    last_modified = time.mktime(time.strptime(last_modified, "%Y-%m-%dT%H:%M:%S.000Z"))
                    im_dict[content['Key']] = last_modified

            # sort by last modified
            sorted_im_dict = dict(sorted(im_dict.items(), key=lambda item: item[1], reverse=True))
            for key in sorted_im_dict.keys():
                im_link = "{}/{}".format(self.end_point, key)
                print(im_link)
                img_list.append(im_link)

        return img_list

    @staticmethod
    def user_name(user_name: str):
        return ''.join(e for e in user_name if e.isalnum() or e == '.')

    @staticmethod
    def get_md5(fname: str) -> str:
        with open(fname, "rb") as f:
            content = f.read()
            md5_hash = md5(content).hexdigest()
        return md5_hash

    @staticmethod
    def get_datetime():
        return time.strftime("%Y%m%d", time.localtime())

    @staticmethod
    def get_video_info(video_path: str):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return {
            'width': width,
            'height': height,
            'duration': duration,
            'fps': fps
        }

    @staticmethod
    def is_video(video_path: str):
        return video_path.lower().strip().endswith(('.mp4', '.mov'))

    @staticmethod
    def get_timestamp():
        return time.strftime("%Y%m%d%H%M%S", time.localtime())


def get_user_info(request: gr.Request):
    try:
        user_agent = request.headers.get("User-Agent")
        user_agent_info = parse(user_agent)
        info = {
            "browser": user_agent_info.browser.family,
            "device": user_agent_info.device.family,
            "os": user_agent_info.os.family,
        }
        logging.info(f"User info: {info}")
        return info
    except Exception as e:
        logging.error(f"Failed to get user info: {str(e)}")
        return {
            "browser": "unknown",
            "device": "unknown", 
            "os": "unknown"
        }


def info_search(user_name: str, selected_date: str, request: gr.Request):
    logging.info(f"Search images for user: {user_name}, date: {selected_date}")

    msg_info = "请输入邮箱."
    template = template = gr.Dropdown([], label="选择日期", interactive=True, visible=False, value=None)

    if user_name == "":
        gr.Warning(msg_info)
        return [], "", template

    try:
        msg_info = f"正在查询 {user_name} 的拼图结果"
        gr.Info(msg_info)
        res = s3client.list_objects(user_name)

        ## no result
        if len(res) == 0:
            logging.info(f"No images found. User: {user_name}")
            template = gr.Dropdown([], label="选择日期", interactive=True, visible=False, value=None)
            res.append(s3client.no_image)

        else:
            # filter images which are not in the selected date
            filtered_res = {}

            logging.debug(f"Selected date: {selected_date}")

            date_list = [img_url.split('/')[-2] for img_url in res]
            date_list = list(set(date_list))
            date_list.sort(reverse=True)

            selected_date = date_list[0] if selected_date is None else selected_date
            template = gr.Dropdown(date_list, label="选择日期", interactive=True, visible=True, value=selected_date)

            # filter images by date
            for img_url in res:
                url_date = img_url.split('/')[-2]
                if url_date == selected_date:
                    filtered_res[img_url] = selected_date

            logging.debug(f"Filtered images: {filtered_res}")
            logging.debug(f"Filtered images by date: {selected_date} - {list(filtered_res.keys())}")
            res = list(filtered_res.keys())

        return res, "查询完成", template

    except Exception as e:
        logging.error(f"Failed to search images: {str(e)}")
        msg_info = f"查询失败，请稍后重试"
        gr.Warning(msg_info)
        
    return [], msg_info, template


def process_video(files: list, user_name: str, request: gr.Request, progress=gr.Progress()):
    if not user_name:
        gr.Warning("请输入邮箱!")
        return [], "请输入邮箱"

    if not files:
        return [], "没有视频文件"

    results = []
    total = len(files)

    progress(0.05, desc="开始处理视频...")

    for idx, file_path in enumerate(files):
        if not s3client.is_video(file_path):
            logging.warning(f"Invalid video format: {file_path}")
            continue

        info_text = f"处理视频 {idx + 1}/{total}"
        progress((idx / total), desc=info_text)

        try:
            # Check if result already exists
            file_md5 = s3client.get_md5(file_path)
            existing_url = s3client.get_url(user_name, file_md5)

            if existing_url:
                logging.info(f"Using existing result for {file_path}")
                results.append(existing_url)
                continue

            # Process new video
            pano = ml_stitch_im_video(file_path, nn_model)
            if pano is not None:
                image_url = s3client.upload_npimg(pano, user_name, file_md5)
                results.append(image_url)
            else:
                logging.error(f"Stitching failed for {file_path}")
                raise Exception("Stitching failed")

        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            gr.Warning(f"视频处理失败: {os.path.basename(file_path)}")
            continue

        yield results, info_text

    final_text = f"已完成 {len(results)}/{total} 个视频处理"
    progress(1.0, desc=final_text)
    gr.Info(final_text)

    s3client.meta_data.update({
        'user_name': user_name,
        'image_url': results,
        **get_user_info(request),
    })
    s3client.upload_metadata(user_name)

    yield results, "拼图完成！"


def upload_video(files: list, user_name: str, progress=gr.Progress()):
    if user_name == "":
        gr.Warning("请输入邮箱")
        return [], "请输入邮箱"

    if not files:
        yield [], "没有视频文件"

    results = []
    total = len(files)
    completed = 0
    info_update = "等待上传..."
    progress(0.05, desc="开始上传...")

    for idx, file_path in enumerate(files):
        info_text = f"开始上传视频 {idx + 1}/{len(files)}"
        progress(idx / len(files), desc=info_text)

        if s3client.is_video(file_path):
            video_url = s3client.upload_file(file_path, user_name)
            completed += 1
            results.append(video_url)
        else:
            logging.warning(f"Invalid video format: {file_path}")
            gr.Warning(f"无效视频格式: {os.path.basename(file_path)}")
            continue

        info_update = f"上传进度 {completed}/{total}"
        progress(completed / total, desc=info_update)
        yield [], info_update

    info_update = f"上传完成！共上传 {completed} 个文件"
    progress(1, desc=info_update)
    gr.Info(info_update)

    s3client.meta_data.update({
        'user_name': user_name,
        'video_url': results,
        'video_res': [s3client.get_video_info(file_path) for file_path in files],
    })

    yield [], info_update


nn_model = NN_Model()
s3client = S3_Client()
meta_data = MetaData()

####### INTERFACE #######
with gr.Blocks(
    title="拼图数据采集",
) as demo:
    gr.Markdown("# 拼图数据采集")
    gr.Markdown("采集前阅读 [数据采集指南](https://clobotics.atlassian.net/wiki/pages/resumedraft.action?draftId=1199439906)")
    gr.Markdown("仅支持 `.mp4, .mov` 视频格式。待视频上传成功后可关闭该页面，之后根据邮箱名查看拼图结果。")

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            with gr.Row():
                user_name   = gr.Textbox(label="邮箱", placeholder="用户名", scale=1)
                domain      = gr.Textbox(
                    label="‎",
                    value="@clobotics.com",
                    interactive=False,
                    scale=1,)

            btn_search  = gr.Button(value="查看结果")
            file_input  = gr.File(label="选择文件", file_count="multiple")

        with gr.Column(scale=2):
            prg_info    = gr.Textbox(label="进度", value="等待上传...")
            dw_date     = gr.Dropdown([], label="选择日期", interactive=True, visible=False)
            gallery     = gr.Gallery(label="上传结果", preview=True)

    file_input.upload(
        fn=upload_video,
        inputs=[file_input, user_name],
        outputs=[gallery, prg_info],
    ).then(
        fn=process_video,
        inputs=[file_input, user_name],
        outputs=[gallery, prg_info],
    )
    # search button
    btn_search.click(
        fn=info_search,
        inputs=[user_name, dw_date],
        outputs=[gallery, prg_info, dw_date],
    )
    # date dropdown
    dw_date.select(
        fn=info_search,
        inputs=[user_name, dw_date],
        outputs=[gallery, prg_info, dw_date],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7878)
