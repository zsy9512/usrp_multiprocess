#!/usr/bin/env python3
"""
端口清理工具 - 清理所有占用指定端口的进程
"""

import subprocess
import sys
import time
import socket

def get_processes_on_port(port):
    """获取占用指定端口的所有进程"""
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        processes = []
        time_wait_count = 0
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port}' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        state = parts[3] if len(parts) > 3 else ""
                        pid = parts[-1]
                        
                        if 'LISTENING' in state:
                            processes.append(pid)
                        elif 'TIME_WAIT' in state:
                            time_wait_count += 1
        
        return list(set(processes)), time_wait_count  # 去重
    except Exception as e:
        print(f"❌ 获取进程列表失败: {e}")
        return [], 0

def check_port_status(host, port):
    """检查端口的详细状态"""
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        listening_processes = []
        time_wait_count = 0
        established_count = 0
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port}' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        state = parts[3] if len(parts) > 3 else ""
                        pid = parts[-1]
                        
                        if 'LISTENING' in state:
                            listening_processes.append(pid)
                        elif 'TIME_WAIT' in state:
                            time_wait_count += 1
                        elif 'ESTABLISHED' in state:
                            established_count += 1
        
        return {
            'listening_processes': list(set(listening_processes)),
            'time_wait_count': time_wait_count,
            'established_count': established_count,
            'is_available': len(listening_processes) == 0
        }
    except Exception as e:
        print(f"❌ 检查端口状态失败: {e}")
        return {
            'listening_processes': [],
            'time_wait_count': 0,
            'established_count': 0,
            'is_available': True
        }

def kill_processes(pids):
    """终止指定的进程"""
    killed = []
    failed = []
    
    for pid in pids:
        try:
            result = subprocess.run(['taskkill', '/PID', pid, '/F'], capture_output=True, text=True)
            if result.returncode == 0:
                killed.append(pid)
                print(f"✅ 已终止进程 (PID: {pid})")
            else:
                failed.append(pid)
                print(f"❌ 终止进程失败 (PID: {pid}): {result.stderr}")
        except Exception as e:
            failed.append(pid)
            print(f"❌ 终止进程异常 (PID: {pid}): {e}")
    
    return killed, failed

def check_port_available(host, port):
    """检查端口是否可用"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0
    except:
        return True

def main():
    print("🧹 USRP DQPSK 端口清理工具")
    print("=" * 40)
    
    # 默认参数
    host = "127.0.0.1"
    port = 50000
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("❌ 端口号必须是整数")
            return
    
    print(f"🔍 检查端口 {host}:{port} 的详细状态...")
    
    # 获取详细的端口状态
    port_status = check_port_status(host, port)
    
    print(f"📊 端口状态详情:")
    print(f"   • LISTENING进程: {len(port_status['listening_processes'])} 个")
    print(f"   • TIME_WAIT连接: {port_status['time_wait_count']} 个")
    print(f"   • ESTABLISHED连接: {port_status['established_count']} 个")
    
    if port_status['listening_processes']:
        print(f"   • 监听进程PID: {', '.join(port_status['listening_processes'])}")
    
    # 检查端口是否真正可用
    if port_status['is_available']:
        print("✅ 端口可用，无需清理")
        if port_status['time_wait_count'] > 0:
            print(f"ℹ️  注意: 还有 {port_status['time_wait_count']} 个TIME_WAIT连接")
            print("   这些连接会在几分钟后自动释放")
        return
    
    print("⚠️  端口被占用，正在查找相关进程...")
    
    # 获取占用端口的进程
    processes, time_wait_count = get_processes_on_port(port)
    
    if not processes:
        print("ℹ️  未找到LISTENING状态的进程")
        if time_wait_count > 0:
            print(f"ℹ️  只有 {time_wait_count} 个TIME_WAIT连接")
            print("💡 TIME_WAIT连接会在2-4分钟后自动释放")
            print("💡 如果需要立即使用端口，可以尝试使用SO_REUSEADDR选项")
        return
    
    print(f"📋 找到 {len(processes)} 个监听进程: {', '.join(processes)}")
    
    # 终止进程
    print("🔧 正在终止进程...")
    killed, failed = kill_processes(processes)
    
    if killed:
        print(f"✅ 成功终止 {len(killed)} 个进程")
    
    if failed:
        print(f"❌ {len(failed)} 个进程终止失败: {', '.join(failed)}")
    
    # 等待并重新检查
    print("⏳ 等待3秒后重新检查端口状态...")
    time.sleep(3)
    
    final_status = check_port_status(host, port)
    if final_status['is_available']:
        print("✅ 端口清理成功！")
        if final_status['time_wait_count'] > 0:
            print(f"ℹ️  还有 {final_status['time_wait_count']} 个TIME_WAIT连接，稍后会自动释放")
    else:
        print("⚠️  端口仍被占用")
        if final_status['listening_processes']:
            print(f"   剩余监听进程: {', '.join(final_status['listening_processes'])}")
        print("💡 尝试手动运行: taskkill /IM python.exe /F")
    
    # 获取占用端口的进程
    processes = get_processes_on_port(port)
    
    if not processes:
        print("ℹ️  未找到占用端口的进程")
        return
    
    print(f"📋 找到 {len(processes)} 个占用端口的进程: {', '.join(processes)}")
    
    # 终止进程
    print("🔧 正在终止进程...")
    killed, failed = kill_processes(processes)
    
    if killed:
        print(f"✅ 成功终止 {len(killed)} 个进程")
    
    if failed:
        print(f"❌ {len(failed)} 个进程终止失败: {', '.join(failed)}")
    
    # 等待并重新检查
    print("⏳ 等待3秒后重新检查端口状态...")
    time.sleep(3)
    
    if check_port_available(host, port):
        print("✅ 端口清理成功！")
    else:
        print("⚠️  端口仍被占用，可能有其他进程")
        print("💡 尝试手动运行: taskkill /IM python.exe /F")

if __name__ == "__main__":
    main()