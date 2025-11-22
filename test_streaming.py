#!/usr/bin/env python3
"""
真正流式输出测试
测试 token-by-token 的流式生成
"""
import requests
import json
import time
import sys


def test_real_streaming(base_url: str = "http://localhost:8000"):
    """测试真正的流式生成"""
    print("🔍 测试真正的流式生成 (token-by-token)...")
    print("=" * 60)
    
    payload = {
        "model": "qwen3",
        "messages": [
            {"role": "user", "content": "请详细介绍一下人工智能的发展历史"}
        ],
        "temperature": 0.7,
        "max_tokens": 200,
        "stream": True
    }
    
    try:
        print(f"📤 发送流式请求...")
        print(f"📝 问题: {payload['messages'][0]['content']}")
        print(f"\n💬 AI 回复:\n")
        print("-" * 60)
        
        start_time = time.time()
        first_token_time = None
        token_count = 0
        total_text = ""
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    
                    if line.startswith('data: '):
                        data_str = line[6:]
                        
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            delta = data['choices'][0]['delta']
                            
                            if 'content' in delta and delta['content']:
                                content = delta['content']
                                total_text += content
                                
                                # 记录第一个token的时间
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    ttft = first_token_time - start_time
                                    print(f"⚡ 首个token延迟: {ttft:.3f}秒\n")
                                
                                # 实时打印
                                print(content, end='', flush=True)
                                token_count += 1
                                
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print("\n" + "-" * 60)
            print(f"\n📊 统计信息:")
            print(f"   总生成时间: {total_time:.3f} 秒")
            if first_token_time:
                print(f"   首token延迟(TTFT): {first_token_time - start_time:.3f} 秒")
            print(f"   生成token数: {token_count}")
            print(f"   总字符数: {len(total_text)}")
            if total_time > 0:
                print(f"   平均速度: {token_count/total_time:.2f} tokens/秒")
            
            print("\n✅ 流式测试完成")
            return True
            
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"   错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ 流式请求失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison(base_url: str = "http://localhost:8000"):
    """对比流式和非流式的性能"""
    print("\n🔍 对比流式 vs 非流式性能...")
    print("=" * 60)
    
    prompt = "请简单介绍一下深度学习"
    
    # 非流式
    print("\n1️⃣ 非流式请求:")
    print("-" * 60)
    
    start = time.time()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "stream": False
        },
        timeout=60
    )
    non_stream_time = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        text = data['choices'][0]['message']['content']
        print(f"⏱️  总时间: {non_stream_time:.3f} 秒")
        print(f"📝 回复: {text[:100]}...")
    
    # 流式
    print("\n2️⃣ 流式请求:")
    print("-" * 60)
    
    start = time.time()
    first_token = None
    
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "stream": True
        },
        stream=True,
        timeout=60
    )
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data['choices'][0]['delta']
                        if 'content' in delta and first_token is None:
                            first_token = time.time() - start
                            print(f"⚡ 首token延迟: {first_token:.3f} 秒")
                            break
                    except:
                        continue
    
    stream_total = time.time() - start
    
    print(f"⏱️  总时间: {stream_total:.3f} 秒")
    
    # 对比
    print("\n📊 对比结果:")
    print("-" * 60)
    print(f"非流式总时间: {non_stream_time:.3f} 秒")
    print(f"流式首token:  {first_token:.3f} 秒 (快 {non_stream_time/first_token:.1f}x)")
    print(f"流式总时间:  {stream_total:.3f} 秒")
    
    print("\n💡 流式的优势: 用户可以更早看到响应开始生成")


def main():
    """运行流式测试"""
    base_url = "http://localhost:8000"
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"🧪 Nano-vLLM 真正流式输出测试")
    print(f"🌐 服务器: {base_url}\n")
    print("=" * 60)
    
    # 检查服务
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ 服务器未就绪")
            return False
    except:
        print("❌ 无法连接到服务器")
        return False
    
    # 测试流式
    success = test_real_streaming(base_url)
    
    # 性能对比
    if success:
        test_comparison(base_url)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

