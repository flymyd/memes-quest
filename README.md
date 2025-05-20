# 图片向量搜索工具

本项目使用 FastAPI、Pymilvus 和 OpenAI Embeddings 构建一个图片向量搜索工具。

## 功能

-   用户可以在指定目录 (`images/`) 添加图片。
-   系统会将图片的文件名（假设包含图片有效信息）通过 OpenAI Embeddings API 向量化。
-   向量数据和图片文件路径存储在 Milvus Lite 中。
-   用户可以通过 FastAPI API 输入文本和期望返回的匹配数量进行查询。
-   API 返回一个包含图片 URL 的数组，FastAPI 同时作为文件服务器提供图片访问。

## 设置

1.  **创建并配置环境:**
    ```bash
    conda create -n memes-quest python==3.11 -y
    pip install -r requirements.txt
    ```

2.  **配置 API 密钥和模型参数:**
    创建 `.env` 文件 (参考 `.env.example` 或直接使用以下内容)，并填入您的 OpenAI API 密钥和自定义模型配置 (如果需要):
    ```env
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY
    # 可选: 如果您使用 OpenAI 兼容的自定义API端点 (例如 Xinference)
    # OPENAI_API_BASE_URL=http://your-custom-api-endpoint/v1 

    # 可选: 指定要使用的嵌入模型名称 (默认为 "text-embedding-ada-002")
    # OPENAI_EMBEDDING_MODEL=bge-large-zh-v1.5

    IMAGE_DIR=images
    COLLECTION_NAME=image_vectors
    
    # 重要: 嵌入向量的维度
    # 必须与您选择的 OPENAI_EMBEDDING_MODEL 输出的维度匹配。
    # - "text-embedding-ada-002" (OpenAI 默认): 1536
    # - "bge-large-zh-v1.5" (示例): 1024
    # - "gte-Qwen2" (示例): 3584
    # 请根据您的模型进行设置。
    EMBEDDING_DIM=1536 

    # Milvus Lite 数据库文件路径 (由应用内部配置，通常不需要在 .env 中设置)
    # MILVUS_URI=./memes_quest.db
    ```

3.  **准备图片:**
    在项目根目录下创建 `images` 文件夹，并将您的图片放入其中。

## 运行

1.  **启动 Milvus Lite:**
    (Pymilvus 会在首次连接时自动处理 Milvus Lite 的下载和启动，如果需要手动管理，请参考 Milvus 文档。)

2.  **启动 FastAPI 应用:**
    ```bash
    uvicorn app.main:app --reload
    ```

## API 端点

-   **POST /index-images/**: 扫描 `IMAGE_DIR` 中的图片，生成嵌入并存入 Milvus。
-   **GET /search/**:
    -   查询参数:
        -   `q` (str): 搜索文本。
        -   `n` (int, optional, default=5): 返回最相似图片的数量。
    -   返回: 图片 URL 列表。
-   **GET /images/{filename}**: 访问图片文件。 