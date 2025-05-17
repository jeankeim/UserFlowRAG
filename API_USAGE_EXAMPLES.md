# Enhanced RAG API 使用示例

## 1. 查询接口

### 请求示例 (cURL)
```bash
curl -X POST "http://localhost:8000/query" \
-H "Authorization: Bearer your_api_token" \
-H "Content-Type: application/json" \
-d '{"question": "妩媚在公司工作几年了？"}'
```

### 请求示例 (Python)
```python
import requests

url = "http://localhost:8000/query"
headers = {
    "Authorization": "Bearer your_api_token",
    "Content-Type": "application/json"
}
data = {
    "question": "妩媚在公司工作几年了？"
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### 成功响应示例
```json
{
    "answer": "妩媚在数据中台工作了4年",
    "sources": [
        {
            "content": "妩媚，在数据中台工作了4年了...",
            "source": "员工信息1.txt",
            "score": 0.95
        }
    ],
    "generated_at": "2025-04-28T23:15:30"
}
```

## 2. 文档上传接口

### 请求示例 (cURL)
```bash
curl -X POST "http://localhost:8000/documents/upload" \
-H "Authorization: Bearer your_api_token" \
-F "file=@/Users/xieming/Desktop/rag_project/data/documents/new.txt"
```

### 请求示例 (Python)
```python
import requests

url = "http://localhost:8000/documents/upload"
headers = {
    "Authorization": "Bearer your_api_token"
}
files = {
    'file': open('员工信息4.txt', 'rb')
}

response = requests.post(url, files=files, headers=headers)
print(response.json())
```

### 成功响应示例
```json
{
    "status": "success",
    "document_id": "doc_12345",
    "processed_chunks": 5,
    "message": "Document processed successfully"
}
```

## 3. 错误处理

### 无效Token示例
```json
{
    "detail": "Invalid token"
}
```

### 无效文件类型示例
```json
{
    "detail": "Unsupported file type. Only PDF, TXT, DOCX allowed"
}
```

### 缺少问题参数示例
```json
{
    "detail": "Question is required"
}
```

## 4. 认证说明
所有API请求需要在Header中包含有效的Bearer Token:
```
Authorization: Bearer your_api_token
