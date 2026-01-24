"""
Command 插件，用于处理命令行输入和执行相应的操作。
"""
from typing import Optional
import inspect
import json
import os
import logging
from lizi_engine.input import input_handler, KeyMap


class Command:
    def __init__(self, controller, commands_file=None):
        self.controller = controller
        if commands_file is None:
            # 默认路径相对于插件文件位置
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            commands_file = os.path.join(plugin_dir, 'commands.json')
        self.commands_file = commands_file
        self.commands = {}
        self.descriptions = {}
        self.no_param_commands = set()
        self.logger = logging.getLogger(__name__)
        self._load_commands()
        # 注册 help 命令
        self.register_command('help', self.list_commands, '显示所有可用指令', no_param=True)

    def _load_commands(self):
        """从外部文件加载命令和描述"""
        if not os.path.exists(self.commands_file):
            raise FileNotFoundError(f"Commands file '{self.commands_file}' not found.")
        try:
            with open(self.commands_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            commands_data = data.get('commands', {})
            descriptions_data = data.get('descriptions', {})
            no_param_commands = data.get('no_param_commands', [])
            for cmd, method_str in commands_data.items():
                try:
                    method = getattr(self.controller, method_str.split('.')[-1])
                    self.commands[cmd] = method
                    self.descriptions[cmd] = descriptions_data.get(cmd, '无描述')
                    self.logger.info(f"Loaded command '{cmd}' -> {method_str}")
                except AttributeError as e:
                    self.logger.error(f"Failed to load command '{cmd}': method '{method_str}' not found on controller. Skipping.")
            self.no_param_commands = set(no_param_commands)
            self.logger.info(f"Loaded {len(self.commands)} commands, {len(self.no_param_commands)} no-param commands.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing commands from '{self.commands_file}': {e}")

    def _convert_arg(self, arg_str: str):
        """尝试将字符串参数转换为适当的类型"""
        try:
            # 尝试转换为 int
            if arg_str.isdigit() or (arg_str.startswith('-') and arg_str[1:].isdigit()):
                return int(arg_str)
            # 尝试转换为 float
            float_val = float(arg_str)
            return float_val
        except ValueError:
            pass
        # 默认返回字符串
        return arg_str

    def execute(self, command_str: str) -> str:
        """执行命令字符串，支持参数"""
        parts = command_str.strip().split()
        if not parts:
            return "错误: 命令为空"

        cmd = parts[0].lstrip('/').lower()
        if cmd not in self.commands:
            return f"错误: 未知命令 '{cmd}'。请使用 'help' 指令查看所有支持指令。"

        args = parts[1:]  # 获取参数
        try:
            func = self.commands[cmd]
            sig = inspect.signature(func)
            param_count = len(sig.parameters)

            if len(args) > 0:
                # 检查函数是否接受参数
                if cmd in self.no_param_commands:
                    return f"错误: 命令 '{cmd}' 不支持参数"
                if param_count == 0:
                    return f"错误: 命令 '{cmd}' 不支持参数"
                if len(args) > param_count:
                    return f"错误: 命令 '{cmd}' 期望 {param_count} 个参数，但提供了 {len(args)} 个"
                if len(args) < param_count:
                    return f"错误: 命令 '{cmd}' 期望 {param_count} 个参数，但提供了 {len(args)} 个"
                # 转换参数类型
                converted_args = [self._convert_arg(arg) for arg in args]
                result = func(*converted_args)
                self.logger.info(f"Executed command '{cmd}' with args {converted_args}")
            else:
                if param_count > 0 and cmd not in self.no_param_commands:
                    return f"错误: 命令 '{cmd}' 需要参数"
                result = func()
                self.logger.info(f"Executed command '{cmd}' without args")

            # 处理返回值
            if result is not None:
                return f"命令 '{cmd}' 执行成功，结果: {result}"
            else:
                return f"命令 '{cmd}' 执行成功"
        except Exception as e:
            self.logger.error(f"Error executing command '{cmd}': {e}")
            return f"错误: 执行命令 '{cmd}' 时发生异常: {type(e).__name__}: {e}"

    def list_commands(self) -> str:
        """列出所有可用命令及其描述"""
        lines = ["可用命令:"]
        for cmd in self.commands.keys():
            desc = self.descriptions.get(cmd, "无描述")
            param_info = ""
            if cmd in self.no_param_commands:
                param_info = " (无参数)"
            else:
                sig = inspect.signature(self.commands[cmd])
                param_count = len(sig.parameters)
                if param_count > 0:
                    param_info = f" (需要 {param_count} 个参数)"
                else:
                    param_info = " (无参数)"
            lines.append(f"- {cmd}{param_info}: {desc}")
        return "\n".join(lines)

    def register_command(self, cmd, method, description, no_param=False):
        """动态注册新命令"""
        self.commands[cmd] = method
        self.descriptions[cmd] = description
        if no_param:
            self.no_param_commands.add(cmd)
        self.logger.info(f"Registered new command '{cmd}' (no_param: {no_param})")


class CommandInputHandler:
    """指令输入处理器，处理指令模式的切换和输入"""

    def __init__(self, command_plugin, history_file='plugins/command_history.json'):
        self.command_plugin = command_plugin
        self.command_mode = False
        self.command_string = ""
        self.cursor_pos = 0
        self.was_pressed = {}
        self.history_file = history_file
        self.history = []
        self.history_index = -1
        self.logger = logging.getLogger(__name__)
        self._load_history()

    def _load_history(self):
        """从文件加载命令历史"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                self.logger.info(f"Loaded {len(self.history)} commands from history")
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Failed to load command history: {e}")
                self.history = []
        else:
            self.history = []

    def _save_history(self):
        """保存命令历史到文件"""
        try:
            # 限制历史记录数量，避免文件过大
            max_history = 100
            if len(self.history) > max_history:
                self.history = self.history[-max_history:]
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except IOError as e:
            self.logger.error(f"Failed to save command history: {e}")

    def toggle_command_mode(self):
        """切换指令输入模式"""
        if not self.command_mode:
            self.command_mode = True
            self.command_string = ""
            self.cursor_pos = 0
            print("[指令模式] 已激活 - 输入指令后按Enter执行，按ESC取消")
        else:
            self.command_mode = False
            print("[指令模式] 已退出")

    def get_toggle_callback(self):
        """获取切换指令模式的回调函数"""
        return self.toggle_command_mode

    def _display_command(self):
        """显示命令字符串，使用终端光标表示位置"""
        print("\033[?25l", end='', flush=True)  # 隐藏终端光标
        print(f"\r\033[K[指令输入] {self.command_string}", end='', flush=True)
        # 移动光标到正确位置
        if self.cursor_pos < len(self.command_string):
            move_left = len(self.command_string) - self.cursor_pos
            print(f"\033[{move_left}D", end='', flush=True)
        print("\033[?25h", end='', flush=True)  # 显示终端光标

    def _handle_command_input(self):
        """处理指令输入，支持引号字符串"""
        # 定义按键到字符的映射
        key_to_char = {
            KeyMap.A: 'a', KeyMap.B: 'b', KeyMap.C: 'c', KeyMap.D: 'd', KeyMap.E: 'e',
            KeyMap.F: 'f', KeyMap.G: 'g', KeyMap.H: 'h', KeyMap.I: 'i', KeyMap.J: 'j',
            KeyMap.K: 'k', KeyMap.L: 'l', KeyMap.M: 'm', KeyMap.N: 'n', KeyMap.O: 'o',
            KeyMap.P: 'p', KeyMap.Q: 'q', KeyMap.R: 'r', KeyMap.S: 's', KeyMap.T: 't',
            KeyMap.U: 'u', KeyMap.V: 'v', KeyMap.W: 'w', KeyMap.X: 'x', KeyMap.Y: 'y',
            KeyMap.Z: 'z',
            KeyMap._0: '0', KeyMap._1: '1', KeyMap._2: '2', KeyMap._3: '3', KeyMap._4: '4',
            KeyMap._5: '5', KeyMap._6: '6', KeyMap._7: '7', KeyMap._8: '8', KeyMap._9: '9',
            KeyMap.SPACE: ' ',
            KeyMap.MINUS: '-',
            KeyMap.EQUAL: '=',
            KeyMap.SEMICOLON: ';',
            KeyMap.APOSTROPHE: "'",
            KeyMap.COMMA: ',',
            KeyMap.PERIOD: '.',
            KeyMap.SLASH: '/',
            # 小键盘映射
            KeyMap.KP_0: '0', KeyMap.KP_1: '1', KeyMap.KP_2: '2', KeyMap.KP_3: '3',
            KeyMap.KP_4: '4', KeyMap.KP_5: '5', KeyMap.KP_6: '6', KeyMap.KP_7: '7',
            KeyMap.KP_8: '8', KeyMap.KP_9: '9',
            KeyMap.KP_DECIMAL: '.',
            KeyMap.KP_DIVIDE: '/',
            KeyMap.KP_MULTIPLY: '*',
            KeyMap.KP_SUBTRACT: '-',
            KeyMap.KP_ADD: '+',
        }

        # 定义Shift按键时的字符映射
        shifted_key_to_char = {
            KeyMap.A: 'A', KeyMap.B: 'B', KeyMap.C: 'C', KeyMap.D: 'D', KeyMap.E: 'E',
            KeyMap.F: 'F', KeyMap.G: 'G', KeyMap.H: 'H', KeyMap.I: 'I', KeyMap.J: 'J',
            KeyMap.K: 'K', KeyMap.L: 'L', KeyMap.M: 'M', KeyMap.N: 'N', KeyMap.O: 'O',
            KeyMap.P: 'P', KeyMap.Q: 'Q', KeyMap.R: 'R', KeyMap.S: 'S', KeyMap.T: 'T',
            KeyMap.U: 'U', KeyMap.V: 'V', KeyMap.W: 'W', KeyMap.X: 'X', KeyMap.Y: 'Y',
            KeyMap.Z: 'Z',
            KeyMap._0: ')', KeyMap._1: '!', KeyMap._2: '@', KeyMap._3: '#', KeyMap._4: '$',
            KeyMap._5: '%', KeyMap._6: '^', KeyMap._7: '&', KeyMap._8: '*', KeyMap._9: '(',
            KeyMap.MINUS: '_',
            KeyMap.EQUAL: '+',
            KeyMap.SEMICOLON: ':',
            KeyMap.APOSTROPHE: '"',
            KeyMap.COMMA: '<',
            KeyMap.PERIOD: '>',
            KeyMap.SLASH: '?',
            # 小键盘映射 (Shift 不影响数字)
            KeyMap.KP_0: '0', KeyMap.KP_1: '1', KeyMap.KP_2: '2', KeyMap.KP_3: '3',
            KeyMap.KP_4: '4', KeyMap.KP_5: '5', KeyMap.KP_6: '6', KeyMap.KP_7: '7',
            KeyMap.KP_8: '8', KeyMap.KP_9: '9',
            KeyMap.KP_DECIMAL: '.',
            KeyMap.KP_DIVIDE: '/',
            KeyMap.KP_MULTIPLY: '*',
            KeyMap.KP_SUBTRACT: '-',
            KeyMap.KP_ADD: '+',
        }

        # 检查Enter键执行指令
        if input_handler.is_key_pressed(KeyMap.ENTER):
            if not self.was_pressed.get(KeyMap.ENTER, False):
                self._execute_command()
            self.was_pressed[KeyMap.ENTER] = True
        else:
            self.was_pressed[KeyMap.ENTER] = False

        # 检查ESC键退出指令模式
        if input_handler.is_key_pressed(KeyMap.ESCAPE):
            if not self.was_pressed.get(KeyMap.ESCAPE, False):
                self.command_mode = False
                print("\n[指令模式] 已退出")
            self.was_pressed[KeyMap.ESCAPE] = True
        else:
            self.was_pressed[KeyMap.ESCAPE] = False

        # 检查Backspace键删除字符
        if input_handler.is_key_pressed(KeyMap.BACKSPACE):
            if not self.was_pressed.get(KeyMap.BACKSPACE, False) and self.cursor_pos > 0:
                self.command_string = self.command_string[:self.cursor_pos-1] + self.command_string[self.cursor_pos:]
                self.cursor_pos -= 1
                self._display_command()
            self.was_pressed[KeyMap.BACKSPACE] = True
        else:
            self.was_pressed[KeyMap.BACKSPACE] = False

        # 检查UP键浏览历史记录（上一条）
        if input_handler.is_key_pressed(KeyMap.UP):
            if not self.was_pressed.get(KeyMap.UP, False) and self.history:
                if self.history_index == -1:
                    # 保存当前输入的命令
                    self.temp_command = self.command_string
                self.history_index = min(self.history_index + 1, len(self.history) - 1)
                self.command_string = self.history[len(self.history) - 1 - self.history_index]
                self.cursor_pos = len(self.command_string)
                self._display_command()
            self.was_pressed[KeyMap.UP] = True
        else:
            self.was_pressed[KeyMap.UP] = False

        # 检查DOWN键浏览历史记录（下一条）
        if input_handler.is_key_pressed(KeyMap.DOWN):
            if not self.was_pressed.get(KeyMap.DOWN, False) and self.history:
                self.history_index = max(self.history_index - 1, -1)
                if self.history_index == -1:
                    # 回到原始输入
                    self.command_string = getattr(self, 'temp_command', "")
                else:
                    self.command_string = self.history[len(self.history) - 1 - self.history_index]
                self.cursor_pos = len(self.command_string)
                self._display_command()
            self.was_pressed[KeyMap.DOWN] = True
        else:
            self.was_pressed[KeyMap.DOWN] = False

        # 检查LEFT键移动光标向左
        if input_handler.is_key_pressed(KeyMap.LEFT):
            if not self.was_pressed.get(KeyMap.LEFT, False):
                self.cursor_pos = max(0, self.cursor_pos - 1)
                self._display_command()
            self.was_pressed[KeyMap.LEFT] = True
        else:
            self.was_pressed[KeyMap.LEFT] = False

        # 检查RIGHT键移动光标向右
        if input_handler.is_key_pressed(KeyMap.RIGHT):
            if not self.was_pressed.get(KeyMap.RIGHT, False):
                self.cursor_pos = min(len(self.command_string), self.cursor_pos + 1)
                self._display_command()
            self.was_pressed[KeyMap.RIGHT] = True
        else:
            self.was_pressed[KeyMap.RIGHT] = False

        # 处理字符输入，支持引号
        in_quotes = False
        quote_char = None
        shift_pressed = input_handler.is_key_pressed(KeyMap.LEFT_SHIFT) or input_handler.is_key_pressed(KeyMap.RIGHT_SHIFT)
        for key, char in key_to_char.items():
            if input_handler.is_key_pressed(key):
                if not self.was_pressed.get(key, False):
                    # 选择映射：Shift按下时使用shifted映射，否则使用普通映射
                    char_map = shifted_key_to_char if shift_pressed else key_to_char
                    char = char_map.get(key, char)
                    # 处理引号开始/结束
                    if char in ('"', "'") and (not in_quotes or char == quote_char):
                        if not in_quotes:
                            in_quotes = True
                            quote_char = char
                        else:
                            in_quotes = False
                            quote_char = None
                    # 在光标位置插入字符
                    self.command_string = self.command_string[:self.cursor_pos] + char + self.command_string[self.cursor_pos:]
                    self.cursor_pos += 1
                    self._display_command()
                self.was_pressed[key] = True
            else:
                self.was_pressed[key] = False

    def _execute_command(self):
        """执行指令"""
        if not self.command_string.strip():
            print("\n[指令] 指令为空")
            self.command_mode = False
            return

        print(f"\n[指令] 执行: {self.command_string}")
        result = self.command_plugin.execute(self.command_string)
        print(f"[指令] 结果: {result}")

        # 如果命令执行成功（不以"错误"开头），添加到历史记录
        if not result.startswith("错误"):
            # 避免重复添加相同的命令
            if not self.history or self.history[-1] != self.command_string:
                self.history.append(self.command_string)
                self._save_history()

        # 重置历史索引
        self.history_index = -1
        if hasattr(self, 'temp_command'):
            delattr(self, 'temp_command')

        self.command_mode = False
