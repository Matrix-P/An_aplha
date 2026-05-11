import re
import pandas as pd
import numpy as np
from functools import wraps

# ---------- 注册向量化算子 ----------
_operators = {}

def register_operator(name):
    """装饰器：注册向量化算子"""
    def decorator(func):
        _operators[name] = func
        return func
    return decorator

# ==================== 基础数学函数 ====================
@register_operator("Abs")
def _abs(x):
    return np.abs(x)

@register_operator("Log")
def _log(x):
    return np.log(x)

@register_operator("Sign")
def _sign(x):
    return np.sign(x)

@register_operator("Sqrt")
def _sqrt(x):
    return np.sqrt(x)

# ==================== 时间序列算子（单资产） ====================
@register_operator("Ref")
def _ref(series, d):
    """Ref(x, d) = x shifted by d periods"""
    d = int(d)
    return series.shift(d)

@register_operator("Delta")
def _delta(series, d):
    """Delta(x, d) = x - Ref(x, d)"""
    d = int(d)
    return series - series.shift(d)

@register_operator("Mean")
def _mean(series, n):
    """滚动均值"""
    n = int(n)
    return series.rolling(n, min_periods=1).mean()

@register_operator("Std")
def _std(series, n):
    """滚动标准差"""
    n = int(n)
    return series.rolling(n, min_periods=1).std()

@register_operator("Sum")
def _sum(series, n):
    """滚动求和"""
    n = int(n)
    return series.rolling(n, min_periods=1).sum()

@register_operator("Max")
def _max(series, n):
    """滚动最大值"""
    n = int(n)
    return series.rolling(n, min_periods=1).max()

@register_operator("Min")
def _min(series, n):
    """滚动最小值"""
    n = int(n)
    return series.rolling(n, min_periods=1).min()

@register_operator("Rank")
def _rank_ts(series, n):
    """
    时间序列排名：过去 n 期的值按升序排名，返回百分比排名（0~1）
    例如：最近 20 天收盘价的排名
    """
    n = int(n)
    # 使用 rolling apply，性能较差，但可接受（或改用 expanding + rank？）
    # 优化：使用列表推导 + 切片
    def _rolling_rank(arr):
        if len(arr) < 2:
            return 0.5
        # 返回最后一个值的百分比排名
        return (arr[-1] > arr[:-1]).sum() / (len(arr)-1)
    return series.rolling(n).apply(_rolling_rank, raw=True)

# ==================== 横截面算子（多资产） ====================
# 这些算子假设输入的 DataFrame 包含 'date' 和 'asset' 列，以及目标列
def _ensure_multi_asset(df):
    """检查是否包含多资产所需列"""
    if 'date' not in df.columns or 'asset' not in df.columns:
        raise ValueError("横截面算子需要 DataFrame 包含 'date' 和 'asset' 列")

@register_operator("CSRank")
def _csrank(series, groupby='date'):
    """横截面排名：每个日期内对资产排名，返回百分比排名（0~1）"""
    # series 可能是 Series 或 DataFrame 的一列
    if hasattr(series, 'groupby'):
        result = series.groupby(groupby).rank(pct=True)
    else:
        # 如果是单列，需要从外部知道分组信息，这里简化：假设传入的是多索引 Series
        result = series.groupby(level=0).rank(pct=True)  # 假设索引第一层是日期
    return result

@register_operator("CSMean")
def _csmean(series, groupby='date'):
    """横截面均值"""
    if hasattr(series, 'groupby'):
        result = series.groupby(groupby).transform('mean')
    else:
        result = series.groupby(level=0).transform('mean')
    return result

@register_operator("CSStd")
def _csstd(series, groupby='date'):
    """横截面标准差"""
    if hasattr(series, 'groupby'):
        result = series.groupby(groupby).transform('std')
    else:
        result = series.groupby(level=0).transform('std')
    return result

# ==================== 滚动相关与协方差 ====================
@register_operator("Corr")
def _corr(x, y, n):
    """滚动相关系数：Corr(x, y, n) = rolling window correlation"""
    n = int(n)
    return x.rolling(n).corr(y)

@register_operator("Cov")
def _cov(x, y, n):
    """滚动协方差：Cov(x, y, n) = rolling window covariance"""
    n = int(n)
    return x.rolling(n).cov(y)

# ==================== 缩放与截断 ====================
@register_operator("Scale")
def _scale(x, a):
    """线性缩放：Scale(x, a) = x * a"""
    return x * float(a)

@register_operator("Clip")
def _clip(x, lower, upper):
    """截断：Clip(x, lower, upper) = clamp x to [lower, upper]"""
    lower = float(lower)
    upper = float(upper)
    return x.clip(lower, upper)

# ==================== 移动平均类 ====================
@register_operator("SMA")
@register_operator("MA")
def _sma(series, window):
    window = int(window)
    return series.rolling(window, min_periods=1).mean()

@register_operator("EMA")
def _ema(series, window):
    window = int(window)
    return series.ewm(span=window, adjust=False).mean()

# 波动率类
@register_operator("RSI")
def _rsi(series, window=14):
    window = int(window)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ==================== 表达式解析与编译 ====================
# （复用之前的解析逻辑，但需要调整变量匹配：支持 $close 或 close 形式）
# 下面给出完整解析器（支持基本四则运算、函数调用、变量）

def _tokenize(expr):
    pattern = r'[+\-*/()]|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][A-Za-z0-9_]*\b|[,]'
    tokens = re.findall(pattern, expr)
    new_tokens = []
    for i, tok in enumerate(tokens):
        if tok == '-' and (i == 0 or tokens[i-1] in ('(', '+', '-', '*', '/', ',')):
            new_tokens.append('~')
        else:
            new_tokens.append(tok)
    return new_tokens

def _parse_expression(tokens, pos=0):
    node, pos = _parse_term(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ('+', '-'):
        op = tokens[pos]
        right, pos = _parse_term(tokens, pos+1)
        node = (op, node, right)
    return node, pos

def _parse_term(tokens, pos):
    node, pos = _parse_factor(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ('*', '/'):
        op = tokens[pos]
        right, pos = _parse_factor(tokens, pos+1)
        node = (op, node, right)
    return node, pos

def _parse_factor(tokens, pos):
    tok = tokens[pos]
    if tok == '(':
        node, pos = _parse_expression(tokens, pos+1)
        if pos < len(tokens) and tokens[pos] == ')':
            return node, pos+1
        raise SyntaxError("括号不匹配")
    elif tok == '~':
        node, pos = _parse_factor(tokens, pos+1)
        return ('neg', node), pos
    elif tok.replace('.', '', 1).isdigit():
        return ('num', float(tok)), pos+1
    elif tok.startswith('$'):
        return ('var', tok[1:]), pos+1
    elif tok.isidentifier():
        # 检查下一个 token 是否为 '('
        if pos+1 < len(tokens) and tokens[pos+1] == '(':
            # 函数调用
            args = []
            arg_pos = pos+2
            while arg_pos < len(tokens) and tokens[arg_pos] != ')':
                arg_node, arg_pos = _parse_expression(tokens, arg_pos)
                args.append(arg_node)
                if arg_pos < len(tokens) and tokens[arg_pos] == ',':
                    arg_pos += 1
            if arg_pos < len(tokens) and tokens[arg_pos] == ')':
                return (tok, args), arg_pos+1
            else:
                raise SyntaxError(f"函数 {tok} 缺少右括号")
        else:
            # 变量（不带 $ 前缀）
            return ('var', tok), pos+1
    raise SyntaxError(f"无法解析的token: {tok}")

def _ast_to_code(node, df_var='df'):
    if isinstance(node, tuple):
        if node[0] in ('+', '-', '*', '/'):
            left = _ast_to_code(node[1], df_var)
            right = _ast_to_code(node[2], df_var)
            return f"({left} {node[0]} {right})"
        elif node[0] == 'neg':
            return f"(-{_ast_to_code(node[1], df_var)})"
        elif node[0] == 'num':
            return str(node[1])
        elif node[0] == 'var':
            return f"{df_var}['{node[1]}']"
        else:  # 函数调用
            func_name = node[0]
            args = [_ast_to_code(arg, df_var) for arg in node[1]]
            if func_name in _operators:
                return f"_operators['{func_name}']({', '.join(args)})"
            else:
                # 未注册函数，直接调用（风险）
                return f"{func_name}({', '.join(args)})"
    else:
        return str(node)

def compile_expression(expr_str, df_var='df'):
    """
    将因子表达式编译为向量化函数。
    返回的函数签名：func(df: pd.DataFrame) -> pd.Series
    """
    tokens = _tokenize(expr_str)
    ast, _ = _parse_expression(tokens)
    code_str = _ast_to_code(ast, df_var)
    func_code = f"""
def calc_factor({df_var}):
    return {code_str}
"""
    namespace = {'pd': pd, 'np': np, '_operators': _operators}
    exec(func_code, namespace)
    return namespace['calc_factor']