import socket
import threading
import queue
import re
import time
import sys
import os

# パスを追加して単体テスト時に config をインポートできるようにする
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import config as cfg
except ImportError:
    print("Error: Could not import config. Make sure config/config.py exists.")
    sys.exit(1)

URL_PATTERN = re.compile(r'https?://\S+')

class TwitchCommentReader:
    def __init__(self, config, output_queue):
        self.config = config
        self.output_queue = output_queue
        self.running = False
        self.thread = None
        self.sock = None

    def start(self):
        if not self.config.TWITCH_COMMENT_ENABLED:
            print("[Twitch] TWITCH_COMMENT_ENABLED is False. Skipping start.")
            return

        if not self.config.TWITCH_CHANNEL_NAME:
            raise ValueError("[Twitch] Error: TWITCH_CHANNEL_NAME is not set in config.")
        if not getattr(self.config, 'TWITCH_BOT_USERNAME', ''):
            raise ValueError("[Twitch] Error: TWITCH_BOT_USERNAME is not set in config.")
        if not self.config.TWITCH_ACCESS_TOKEN:
            raise ValueError("[Twitch] Error: TWITCH_ACCESS_TOKEN is not set in config.")

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"[Twitch] Started reader for channel: {self.config.TWITCH_CHANNEL_NAME}")

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        print("[Twitch] Stopped Twitch comment reader.")

    def _run(self):
        while self.running:
            try:
                self.sock = socket.socket()
                self.sock.settimeout(300) # Keepalive timeout
                self.sock.connect(('irc.chat.twitch.tv', 6667))
                
                token = self.config.TWITCH_ACCESS_TOKEN
                if not token.startswith('oauth:'):
                    token = f"oauth:{token}"
                
                # Nickname using TWITCH_BOT_USERNAME
                nick = self.config.TWITCH_BOT_USERNAME.lower()
                channel = f"#{self.config.TWITCH_CHANNEL_NAME.lower()}"
                
                self.sock.send(f"PASS {token}\r\n".encode('utf-8'))
                self.sock.send(f"NICK {nick}\r\n".encode('utf-8'))
                self.sock.send(f"JOIN {channel}\r\n".encode('utf-8'))
                
                buffer = ""
                while self.running:
                    try:
                        resp = self.sock.recv(2048).decode('utf-8', errors='ignore')
                        if not resp:
                            break
                        
                        if getattr(self.config, 'TWITCH_DEBUG_LOG', False):
                            print(f"[Twitch Debug] {resp.strip()}")
                            
                        buffer += resp
                        while "\r\n" in buffer:
                            line, buffer = buffer.split("\r\n", 1)
                            self._process_line(line)
                    except socket.timeout:
                        # Continue to keep the loop going if just timed out
                        continue
                    except socket.error:
                        break
            except Exception as e:
                if self.running:
                    print(f"[Twitch] Connection error: {e}")
            
            if self.running:
                print("[Twitch] Disconnected. Reconnecting in 5 seconds...")
                time.sleep(5)

    def _process_line(self, line):
        if line.startswith('PING'):
            self.sock.send(f"PONG {line.split()[1]}\r\n".encode('utf-8'))
            return

        match = re.match(r'^:([^!]+)!.* PRIVMSG #[^ ]+ :(.*)', line)
        if match:
            username = match.group(1)
            message = match.group(2)
            self._handle_message(username, message)

    def _handle_message(self, username, message):
        message = message.strip()
        
        # Filter rules
        if not message:
            return
            
        if len(message) > self.config.COMMENT_MAX_LENGTH:
            return
            
        if message.startswith(self.config.COMMENT_IGNORE_PREFIXES):
            return
            
        if self.config.COMMENT_IGNORE_URL and URL_PATTERN.search(message):
            return

        # Output to console
        print(f"[COMMENT] {username}: {message}")
        
        # Output to queue
        if self.output_queue is not None:
            self.output_queue.put({"username": username, "message": message})


if __name__ == "__main__":
    # For standalone testing
    print("--- Twitch Comment Reader Standalone Test ---")
    
    # 起動時に設定不足の場合は分かりやすいエラーを出すこと
    missing_configs = []
    if not getattr(cfg, 'TWITCH_CHANNEL_NAME', ''):
        missing_configs.append("TWITCH_CHANNEL_NAME")
    if not getattr(cfg, 'TWITCH_BOT_USERNAME', ''):
        missing_configs.append("TWITCH_BOT_USERNAME")
    if not getattr(cfg, 'TWITCH_ACCESS_TOKEN', ''):
        missing_configs.append("TWITCH_ACCESS_TOKEN")
        
    if missing_configs:
        print("Error: Missing required Twitch configuration.")
        print(f"Please set the following variables in config/config.py: {', '.join(missing_configs)}")
        sys.exit(1)

    # テスト時は強制的に有効化
    cfg.TWITCH_COMMENT_ENABLED = True
    
    q = queue.Queue()
    reader = TwitchCommentReader(cfg, q)
    
    try:
        reader.start()
        print("Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        reader.stop()
        sys.exit(0)
