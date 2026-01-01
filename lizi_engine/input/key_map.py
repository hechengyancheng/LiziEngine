"""
键盘映射 - 定义常用键的映射
"""
# Use GLFW key codes as they are compatible with Dear PyGui
# GLFW key codes are standard ASCII values for most keys


class KeyMap:
    """键盘映射类，提供常用键的常量定义"""

    # 特殊键
    UNKNOWN = -1
    SPACE = 32  # GLFW.KEY_SPACE
    APOSTROPHE = 39  # GLFW.KEY_APOSTROPHE
    COMMA = 44  # GLFW.KEY_COMMA
    MINUS = 45  # GLFW.KEY_MINUS
    PERIOD = 46  # GLFW.KEY_PERIOD
    SLASH = 47  # GLFW.KEY_SLASH
    SEMICOLON = 59  # GLFW.KEY_SEMICOLON
    EQUAL = 61  # GLFW.KEY_EQUAL

    # 数字键
    _0 = 48  # GLFW.KEY_0
    _1 = 49  # GLFW.KEY_1
    _2 = 50  # GLFW.KEY_2
    _3 = 51  # GLFW.KEY_3
    _4 = 52  # GLFW.KEY_4
    _5 = 53  # GLFW.KEY_5
    _6 = 54  # GLFW.KEY_6
    _7 = 55  # GLFW.KEY_7
    _8 = 56  # GLFW.KEY_8
    _9 = 57  # GLFW.KEY_9

    # 字母键
    A = 65  # GLFW.KEY_A
    B = 66  # GLFW.KEY_B
    C = 67  # GLFW.KEY_C
    D = 68  # GLFW.KEY_D
    E = 69  # GLFW.KEY_E
    F = 70  # GLFW.KEY_F
    G = 71  # GLFW.KEY_G
    H = 72  # GLFW.KEY_H
    I = 73  # GLFW.KEY_I
    J = 74  # GLFW.KEY_J
    K = 75  # GLFW.KEY_K
    L = 76  # GLFW.KEY_L
    M = 77  # GLFW.KEY_M
    N = 78  # GLFW.KEY_N
    O = 79  # GLFW.KEY_O
    P = 80  # GLFW.KEY_P
    Q = 81  # GLFW.KEY_Q
    R = 82  # GLFW.KEY_R
    S = 83  # GLFW.KEY_S
    T = 84  # GLFW.KEY_T
    U = 85  # GLFW.KEY_U
    V = 86  # GLFW.KEY_V
    W = 87  # GLFW.KEY_W
    X = 88  # GLFW.KEY_X
    Y = 89  # GLFW.KEY_Y
    Z = 90  # GLFW.KEY_Z

    # 功能键
    F1 = 290  # GLFW.KEY_F1
    F2 = 291  # GLFW.KEY_F2
    F3 = 292  # GLFW.KEY_F3
    F4 = 293  # GLFW.KEY_F4
    F5 = 294  # GLFW.KEY_F5
    F6 = 295  # GLFW.KEY_F6
    F7 = 296  # GLFW.KEY_F7
    F8 = 297  # GLFW.KEY_F8
    F9 = 298  # GLFW.KEY_F9
    F10 = 299  # GLFW.KEY_F10
    F11 = 300  # GLFW.KEY_F11
    F12 = 301  # GLFW.KEY_F12

    # 方向键
    UP = 265  # GLFW.KEY_UP
    DOWN = 264  # GLFW.KEY_DOWN
    LEFT = 263  # GLFW.KEY_LEFT
    RIGHT = 262  # GLFW.KEY_RIGHT

    # 特殊键
    LEFT_SHIFT = 340  # GLFW.KEY_LEFT_SHIFT
    RIGHT_SHIFT = 344  # GLFW.KEY_RIGHT_SHIFT
    LEFT_CONTROL = 341  # GLFW.KEY_LEFT_CONTROL
    RIGHT_CONTROL = 345  # GLFW.KEY_RIGHT_CONTROL
    LEFT_ALT = 342  # GLFW.KEY_LEFT_ALT
    RIGHT_ALT = 346  # GLFW.KEY_RIGHT_ALT
    LEFT_SUPER = 343  # GLFW.KEY_LEFT_SUPER
    RIGHT_SUPER = 347  # GLFW.KEY_RIGHT_SUPER
    TAB = 258  # GLFW.KEY_TAB
    ENTER = 257  # GLFW.KEY_ENTER
    BACKSPACE = 259  # GLFW.KEY_BACKSPACE
    INSERT = 260  # GLFW.KEY_INSERT
    DELETE = 261  # GLFW.KEY_DELETE
    PAGE_UP = 266  # GLFW.KEY_PAGE_UP
    PAGE_DOWN = 267  # GLFW.KEY_PAGE_DOWN
    HOME = 268  # GLFW.KEY_HOME
    END = 269  # GLFW.KEY_END
    CAPS_LOCK = 280  # GLFW.KEY_CAPS_LOCK
    SCROLL_LOCK = 281  # GLFW.KEY_SCROLL_LOCK
    NUM_LOCK = 282  # GLFW.KEY_NUM_LOCK
    PRINT_SCREEN = 283  # GLFW.KEY_PRINT_SCREEN
    PAUSE = 284  # GLFW.KEY_PAUSE
    ESCAPE = 256  # GLFW.KEY_ESCAPE

    # 小键盘
    KP_0 = 320  # GLFW.KEY_KP_0
    KP_1 = 321  # GLFW.KEY_KP_1
    KP_2 = 322  # GLFW.KEY_KP_2
    KP_3 = 323  # GLFW.KEY_KP_3
    KP_4 = 324  # GLFW.KEY_KP_4
    KP_5 = 325  # GLFW.KEY_KP_5
    KP_6 = 326  # GLFW.KEY_KP_6
    KP_7 = 327  # GLFW.KEY_KP_7
    KP_8 = 328  # GLFW.KEY_KP_8
    KP_9 = 329  # GLFW.KEY_KP_9
    KP_DECIMAL = 330  # GLFW.KEY_KP_DECIMAL
    KP_DIVIDE = 331  # GLFW.KEY_KP_DIVIDE
    KP_MULTIPLY = 332  # GLFW.KEY_KP_MULTIPLY
    KP_SUBTRACT = 333  # GLFW.KEY_KP_SUBTRACT
    KP_ADD = 334  # GLFW.KEY_KP_ADD
    KP_ENTER = 335  # GLFW.KEY_KP_ENTER
    KP_EQUAL = 336  # GLFW.KEY_KP_EQUAL

    # 修饰键
    MOD_SHIFT = 1  # GLFW.MOD_SHIFT
    MOD_CONTROL = 2  # GLFW.MOD_CONTROL
    MOD_ALT = 4  # GLFW.MOD_ALT
    MOD_SUPER = 8  # GLFW.MOD_SUPER

    @staticmethod
    def get_key_name(key: int) -> str:
        """获取键名

        Args:
            key: GLFW键码

        Returns:
            str: 键名
        """
        key_names = {
            KeyMap.SPACE: "Space",
            KeyMap.APOSTROPHE: "'",
            KeyMap.COMMA: ",",
            KeyMap.MINUS: "-",
            KeyMap.PERIOD: ".",
            KeyMap.SLASH: "/",
            KeyMap.SEMICOLON: ";",
            KeyMap.EQUAL: "=",
            KeyMap._0: "0",
            KeyMap._1: "1",
            KeyMap._2: "2",
            KeyMap._3: "3",
            KeyMap._4: "4",
            KeyMap._5: "5",
            KeyMap._6: "6",
            KeyMap._7: "7",
            KeyMap._8: "8",
            KeyMap._9: "9",
            KeyMap.A: "A",
            KeyMap.B: "B",
            KeyMap.C: "C",
            KeyMap.D: "D",
            KeyMap.E: "E",
            KeyMap.F: "F",
            KeyMap.G: "G",
            KeyMap.H: "H",
            KeyMap.I: "I",
            KeyMap.J: "J",
            KeyMap.K: "K",
            KeyMap.L: "L",
            KeyMap.M: "M",
            KeyMap.N: "N",
            KeyMap.O: "O",
            KeyMap.P: "P",
            KeyMap.Q: "Q",
            KeyMap.R: "R",
            KeyMap.S: "S",
            KeyMap.T: "T",
            KeyMap.U: "U",
            KeyMap.V: "V",
            KeyMap.W: "W",
            KeyMap.X: "X",
            KeyMap.Y: "Y",
            KeyMap.Z: "Z",
            KeyMap.F1: "F1",
            KeyMap.F2: "F2",
            KeyMap.F3: "F3",
            KeyMap.F4: "F4",
            KeyMap.F5: "F5",
            KeyMap.F6: "F6",
            KeyMap.F7: "F7",
            KeyMap.F8: "F8",
            KeyMap.F9: "F9",
            KeyMap.F10: "F10",
            KeyMap.F11: "F11",
            KeyMap.F12: "F12",
            KeyMap.UP: "Up",
            KeyMap.DOWN: "Down",
            KeyMap.LEFT: "Left",
            KeyMap.RIGHT: "Right",
            KeyMap.LEFT_SHIFT: "Left Shift",
            KeyMap.RIGHT_SHIFT: "Right Shift",
            KeyMap.LEFT_CONTROL: "Left Control",
            KeyMap.RIGHT_CONTROL: "Right Control",
            KeyMap.LEFT_ALT: "Left Alt",
            KeyMap.RIGHT_ALT: "Right Alt",
            KeyMap.LEFT_SUPER: "Left Super",
            KeyMap.RIGHT_SUPER: "Right Super",
            KeyMap.TAB: "Tab",
            KeyMap.ENTER: "Enter",
            KeyMap.BACKSPACE: "Backspace",
            KeyMap.INSERT: "Insert",
            KeyMap.DELETE: "Delete",
            KeyMap.PAGE_UP: "Page Up",
            KeyMap.PAGE_DOWN: "Page Down",
            KeyMap.HOME: "Home",
            KeyMap.END: "End",
            KeyMap.CAPS_LOCK: "Caps Lock",
            KeyMap.SCROLL_LOCK: "Scroll Lock",
            KeyMap.NUM_LOCK: "Num Lock",
            KeyMap.PRINT_SCREEN: "Print Screen",
            KeyMap.PAUSE: "Pause",
            KeyMap.ESCAPE: "Escape",
            KeyMap.KP_0: "Keypad 0",
            KeyMap.KP_1: "Keypad 1",
            KeyMap.KP_2: "Keypad 2",
            KeyMap.KP_3: "Keypad 3",
            KeyMap.KP_4: "Keypad 4",
            KeyMap.KP_5: "Keypad 5",
            KeyMap.KP_6: "Keypad 6",
            KeyMap.KP_7: "Keypad 7",
            KeyMap.KP_8: "Keypad 8",
            KeyMap.KP_9: "Keypad 9",
            KeyMap.KP_DECIMAL: "Keypad Decimal",
            KeyMap.KP_DIVIDE: "Keypad Divide",
            KeyMap.KP_MULTIPLY: "Keypad Multiply",
            KeyMap.KP_SUBTRACT: "Keypad Subtract",
            KeyMap.KP_ADD: "Keypad Add",
            KeyMap.KP_ENTER: "Keypad Enter",
            KeyMap.KP_EQUAL: "Keypad Equal"
        }
        return key_names.get(key, "Unknown")
