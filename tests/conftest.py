import os
import sys
import types

# Ensure project root is on sys.path so tests can import the `src` package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 최소 더미 모듈들을 삽입하여 외부 의존성 없이도 import가 가능하도록 합니다.
DUMMY_MODULES = [
    'dotenv', 'pyupbit', 'ccxt', 'pandas', 'pandas_ta', 'schedule', 'requests', 'smtplib', 'email', 'tweepy', 'praw'
]

for name in DUMMY_MODULES:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        # dotenv 사용 시 load_dotenv 함수가 필요합니다.
        if name == 'dotenv':
            def load_dotenv(path=None):
                return None
            mod.load_dotenv = load_dotenv
        # pandas는 일부 코드에서 DataFrame 존재만 확인하므로 간단한 대체 클래스를 추가합니다.
        if name == 'pandas':
            class DummyDataFrame:
                def __init__(self, *args, **kwargs):
                    pass
                @property
                def empty(self):
                    return False
                def to_csv(self, *args, **kwargs):
                    return None
            mod.DataFrame = DummyDataFrame
            # pandas has 'to_datetime' or other attributes sometimes used; provide safe fallbacks
            def to_datetime(x, *args, **kwargs):
                return x
            mod.to_datetime = to_datetime
            def concat(objs, *args, **kwargs):
                return objs[0] if objs else DummyDataFrame()
            mod.concat = concat
        sys.modules[name] = mod

# Provide a minimal numpy stub to satisfy imports in tests
if 'numpy' not in sys.modules:
    import types as _types
    np = _types.ModuleType('numpy')
    # minimal functions used by utils
    def ndarray(x):
        return x
    np.ndarray = ndarray
    def asarray(x, *args, **kwargs):
        return x
    np.asarray = asarray
    def nan_to_num(x, *args, **kwargs):
        return x
    np.nan_to_num = nan_to_num
    np.nan = float('nan')
    # add isscalar used by pytest approx
    def isscalar(x):
        return isinstance(x, (int, float, complex))
    np.isscalar = isscalar
    # provide common numpy dtypes used by other libs/tests
    np.bool_ = bool
    np.float64 = float
    np.int64 = int
    sys.modules['numpy'] = np

# minimal matplotlib stub
if 'matplotlib' not in sys.modules:
    import types as _types
    mpl = _types.ModuleType('matplotlib')
    plt = _types.ModuleType('matplotlib.pyplot')
    def plot(*args, **kwargs):
        return None
    plt.plot = plot
    mpl.pyplot = plt
    sys.modules['matplotlib'] = mpl
    sys.modules['matplotlib.pyplot'] = plt
