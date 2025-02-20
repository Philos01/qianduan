from concurrent.futures.thread import ThreadPoolExecutor
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from models.demo import demo_func
from models.fusion import SARF_MsPan_Fuse

app = Flask(__name__)
CORS(app)
pool = ThreadPoolExecutor(max_workers=20, thread_name_prefix='geo_client')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/algorithm1', methods=['POST'])
def algorithm1():
    try:
        # 检查文件是否在请求中
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': '两个文件都必须上传'}), 400

        # 获取上传的文件
        file1 = request.files['image1']
        file2 = request.files['image2']

        # Validate file types
        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # 保存文件
        file1_path = os.path.join('uploads', file1.filename)
        file2_path = os.path.join('uploads', file2.filename)
        file1.save(file1_path)
        file2.save(file2_path)
        message_list = [file1_path, file2_path]
        # 创建数据字典
        data = {
            'file1': file1.filename,
            'file2': file2.filename,
            'job_id': request.form.get('job_id', 1),
            'type': request.form.get('type', 'test'),
            'message_list': message_list,
            'GroundTruth1': request.form.get('GroundTruth1'),
            'GroundTruth2': request.form.get('GroundTruth2'),
            'otherInfo': request.form.get('otherInfo')
        }

        print(f"Received files: {file1.filename}, {file2.filename}")
        print(f"Received other data: {data}")

        # 提交任务并添加日志
        future = pool.submit(demo_func, data['job_id'],
                             data['type'],
                             data['message_list'],
                             data['GroundTruth1'],
                             data['GroundTruth2'],
                             data['otherInfo'])

        # 等待结果并返回
        result = future.result()  # 获取任务的返回值
        # print(f"Task {data['job_id']} completed with result: {result}")
        return jsonify({"message": f"Job completed with ID: {data['job_id']}", "result": result}), 200


    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route('/algorithm2', methods=['POST'])
def algorithm2():
    try:
        # 检查文件是否在请求中
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': '两个文件都必须上传'}), 400

        # 获取上传的文件
        file1 = request.files['image1']
        file2 = request.files['image2']

        # Validate file types
        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # 保存文件
        file1_path = os.path.join('uploads', file1.filename)
        file2_path = os.path.join('uploads', file2.filename)
        file1.save(file1_path)
        file2.save(file2_path)
        # 创建数据字典
        data = {
            'job_id': 2,
            'file1': file1_path,
            'file2': file2_path,
            'save_dir': r'F:\file_zxt\group_alg\geo_client\geo_client\mspanfusion-pairs/',
            'file_name': 'fusion_image',
            'lamda0': 0.3
        }

        print(f"Received files: {file1.filename}, {file2.filename}")
        print(f"Received other data: {data}")

        # 提交任务并添加日志
        future = pool.submit(SARF_MsPan_Fuse, data['job_id'],
                             data['file1'],
                             data['file2'],
                             data['save_dir'],
                             data['file_name'],
                             data['lamda0'])

        # 等待结果并返回
        result = future.result()  # 获取任务的返回值
        # print(f"Task {data['job_id']} completed with result: {result}")
        return jsonify({"message": f"Job completed with ID: {data['job_id']}", "result": result}), 200


    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(threaded=True)
