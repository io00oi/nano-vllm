#!/usr/bin/env python3
"""
API 测试脚本
测试 Nano-vLLM API Server 的各个功能
"""
import requests
import json
import sys


def test_health(base_url: str = "http://localhost:8000"):
    """测试健康检查"""
    print("🔍 测试健康检查...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查成功: {data}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


def test_models(base_url: str = "http://localhost:8000"):
    """测试模型列表"""
    print("\n🔍 测试模型列表...")
    try:
        response = requests.get(f"{base_url}/v1/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 模型列表: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def test_chat_completion(base_url: str = "http://localhost:8000"):
    """测试聊天完成（非流式）"""
    print("\n🔍 测试聊天完成（非流式）...")
    
    payload = {
        "model": "qwen3",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False
    }
    
    try:
        print(f"📤 发送请求: {json.dumps(payload, ensure_ascii=False)}")
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 响应成功:")
            print(f"   ID: {data['id']}")
            print(f"   回复: {data['choices'][0]['message']['content']}")
            print(f"   Token 使用: {data['usage']}")
            return True
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"   错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def test_chat_completion_stream(base_url: str = "http://localhost:8000"):
    """测试聊天完成（流式）"""
    print("\n🔍 测试聊天完成（流式）...")
    
    payload = {
        "model": "qwen3",
        "messages": [
            {"role": "user", "content": "请写一首短诗"}
        ],
        "temperature": 0.8,
        "max_tokens": 100,
        "stream": True
    }
    
    try:
        print(f"📤 发送流式请求...")
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=60
        )
        
        if response.status_code == 200:
            print("✅ 开始接收流式响应:")
            print("📝 ", end="", flush=True)
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # 移除 'data: ' 前缀
                        
                        if data_str == '[DONE]':
                            print("\n✅ 流式响应完成")
                            break
                        
                        try:
                            data = json.loads(data_str)
                            delta = data['choices'][0]['delta']
                            if 'content' in delta:
                                print(delta['content'], end="", flush=True)
                        except json.JSONDecodeError:
                            continue
            
            return True
        else:
            print(f"❌ 流式请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 流式请求失败: {e}")
        return False


def test_completion(base_url: str = "http://localhost:8000"):
    """测试文本完成"""
    print("\n🔍 测试文本完成...")
    
    payload = {
        "model": "qwen3",
        "prompt": "人工智能的未来是",
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 响应成功:")
            print(f"   生成文本: {data['choices'][0]['text']}")
            return True
        else:
            print(f"❌ 请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def main():
    """运行所有测试"""
    base_url = "http://localhost:8000"
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"🧪 开始测试 Nano-vLLM API Server")
    print(f"🌐 服务器地址: {base_url}\n")
    print("=" * 60)
    
    results = []
    
    # 1. 健康检查
    results.append(("健康检查", test_health(base_url)))
    
    # 2. 模型列表
    results.append(("模型列表", test_models(base_url)))
    
    # 3. 聊天完成（非流式）
    results.append(("聊天完成(非流式)", test_chat_completion(base_url)))
    
    # 4. 聊天完成（流式）
    results.append(("聊天完成(流式)", test_chat_completion_stream(base_url)))
    
    # 5. 文本完成
    results.append(("文本完成", test_completion(base_url)))
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结:")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name:20s} {status}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print("=" * 60)
    print(f"总计: {passed}/{total} 通过\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

