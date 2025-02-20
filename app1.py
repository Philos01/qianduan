from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/algorithm1', methods=['POST'])
def upload():
    # 从请求体中获取 JSON 数据
    data = request.get_json()
    if not data or 'image1' not in data or 'image2' not in data:
        return jsonify({"error": "Invalid input"}), 400

    # 提取图片地址
    image1_url = data['image1']
    image2_url = data['image2']

    # 这里你可以对图片地址进行处理
    print(f"Received Image 1 URL: {image1_url}")
    print(f"Received Image 2 URL: {image2_url}")

    # 返回响应
    return jsonify({"message": "Images received successfully"}), 200


if __name__ == '__main__':
    app.run(debug=True)
