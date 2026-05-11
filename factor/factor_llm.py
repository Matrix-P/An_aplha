# llm_factor_optimizer.py

import json
import logging
from typing import Any, Dict, List, Optional, TypedDict
import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np

from langgraph.graph import END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# 导入本项目已有的模块
from .factor_compute import compute_factors_for_pool
from .ic_compute import calc_ic_series, calc_icir

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- 计算收益率 ----------
def compute_return_factor(pro, stock_pool, start_date, end_date) -> pd.DataFrame:
    """
    计算收益率因子：次日收益率 = (Ref($close, -1) - $close) / $close
    返回长格式 DataFrame，包含 trade_date, stock, ret_t1 列
    """
    logger.info("正在计算收益率因子...")
    ret_expr = "(Ref($close, -1) - $close) / $close"
    ret_dict = {"ret_t1": ret_expr}
    df_ret = compute_factors_for_pool(
        pro=pro,
        factor_dict=ret_dict,
        stock_pool=stock_pool,
        start_date=start_date,
        end_date=end_date,
    )
    df_ret = df_ret['ret_t1']
    logger.info(f"收益率因子计算完成，共 {len(df_ret)} 行")
    return df_ret

# ---------- 单个因子评估（模块级工具函数）----------
def _evaluate_factor(factor_name: str, factor_expr: str, factor_series: pd.Series, ret_df: pd.DataFrame, trim_quantile: tuple, min_samples: int) -> Optional[Dict[str, Any]]:
    """评估单个因子，返回包含 IC综合 的完整指标字典"""
    ic_df = calc_ic_series(
        factor=factor_series,
        forward_ret=ret_df,
        by_date=True,
        date_col=None,
        trim_quantile=trim_quantile,
        min_samples=min_samples,
    )
    if ic_df.empty:
        logger.warning(f"因子 {factor_name} 的 IC 序列为空")
        return None

    metrics = calc_icir(ic_df)
    if pd.isna(metrics.get('ICIR', np.nan)) and pd.isna(metrics.get('RankICIR', np.nan)):
        return None

    # IC综合 = |mean| × |ICIR|，惩罚低均值低波动的情况
    ic_comp = abs(metrics['mean_IC']) * abs(metrics['ICIR'])
    rank_comp = abs(metrics['mean_RankIC']) * abs(metrics['RankICIR'])
    ic_composite = 0.5 * ic_comp + 0.5 * rank_comp

    return {
        "名称": factor_name,
        "表达式": factor_expr,
        "IC综合": ic_composite,
        "ICIR": metrics['ICIR'],
        "RankICIR": metrics['RankICIR'],
        "mean_IC": metrics['mean_IC'],
        "mean_RankIC": metrics['mean_RankIC'],
        "std_IC": metrics['std_IC'],
        "std_RankIC": metrics['std_RankIC'],
    }


class FactorOptimizationState(TypedDict):
    current_batch: int
    total_factors_needed: int
    generated_factors: List[Dict[str, str]]
    evaluated_factors: List[Dict[str, Any]]
    factor_table: pd.DataFrame
    status: str
    error: Optional[str]
    final_result: Optional[Dict[str, Any]]


class FactorOptimizer:
    def __init__(
        self,
        pro,                          # Tushare pro 对象
        stock_pool: List[str],        # 股票代码列表，如 ['600519.SH', ...]
        start_date: str,              # 因子计算起始日期 'YYYY-MM-DD'
        end_date: str,                # 因子计算结束日期
        api_key: Optional[str] = None,
        llm_model_uri: str = "https://api.deepseek.com",
        llm_model: str = "deepseek-chat",
        batch_size: int = 20,
        total_factors_needed: int = 100,
        top_n: int = 20,
        trim_quantile: tuple = (0.01, 0.99),
        min_samples: int = 5,
    ):
        self.pro = pro
        self.stock_pool = stock_pool
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        self.total_factors_needed = total_factors_needed
        self.top_n = top_n
        self.trim_quantile = trim_quantile
        self.min_samples = min_samples

        # 初始化LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            openai_api_key=api_key,
            openai_api_base=llm_model_uri,
            temperature=0.7,
        )

        # 预先计算收益率因子（t+1收益率）并缓存
        self.ret_df = self._compute_return_factor()

        # 创建工作流
        self.workflow = self._create_workflow()
        logger.info("因子优化器初始化完成")

    # ---------- 收益率计算（使用因子表达式，避免额外数据获取） ----------
    def _compute_return_factor(self) -> pd.DataFrame:
        logger.info("正在计算收益率因子...")
        ret_expr = "(Ref($close, -1) - $close) / $close"
        ret_dict = {"ret_t1": ret_expr}
        df_ret = compute_factors_for_pool(
            pro=self.pro,
            factor_dict=ret_dict,
            stock_pool=self.stock_pool,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        df_ret = df_ret['ret_t1']
        logger.info(f"收益率因子计算完成，共 {len(df_ret)} 行")
        return df_ret

    # ---------- 单个因子评估 ----------
    def _evaluate_factor(self, factor_name: str, factor_expr: str, factor_series: pd.Series) -> Optional[Dict[str, Any]]:
        """评估单个因子，返回包含 IC综合 的完整指标字典"""
        ic_df = calc_ic_series(
            factor=factor_series,
            forward_ret=self.ret_df,
            by_date=True,
            date_col=None,
            trim_quantile=self.trim_quantile,
            min_samples=self.min_samples,
        )
        if ic_df.empty:
            logger.warning(f"因子 {factor_name} 的 IC 序列为空")
            return None

        metrics = calc_icir(ic_df)
        if pd.isna(metrics.get('ICIR', np.nan)) and pd.isna(metrics.get('RankICIR', np.nan)):
            return None

        # IC综合 = |mean| × |ICIR|，惩罚低均值低波动的情况
        ic_comp = abs(metrics['mean_IC']) * abs(metrics['ICIR'])
        rank_comp = abs(metrics['mean_RankIC']) * abs(metrics['RankICIR'])
        ic_composite = 0.5 * ic_comp + 0.5 * rank_comp

        return {
            "名称": factor_name,
            "表达式": factor_expr,
            "IC综合": ic_composite,
            "ICIR": metrics['ICIR'],
            "RankICIR": metrics['RankICIR'],
            "mean_IC": metrics['mean_IC'],
            "mean_RankIC": metrics['mean_RankIC'],
            "std_IC": metrics['std_IC'],
            "std_RankIC": metrics['std_RankIC'],
        }
    

    # ---------- LangGraph 工作流定义 ----------
    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(FactorOptimizationState)

        workflow.add_node("start", self._start_node)
        workflow.add_node("generate_factors", self._generate_factors_node)
        workflow.add_node("evaluate_factors", self._evaluate_factors_node)
        workflow.add_node("collect_results", self._collect_results_node)
        workflow.add_node("end", self._end_node)

        workflow.set_entry_point("start")
        workflow.add_edge("start", "generate_factors")
        workflow.add_edge("generate_factors", "evaluate_factors")
        workflow.add_edge("evaluate_factors", "collect_results")
        workflow.add_conditional_edges(
            "collect_results",
            self._should_continue,
            {"continue": "generate_factors", "end": "end"},
        )
        workflow.add_edge("end", END)
        return workflow

    def _start_node(self, state: FactorOptimizationState) -> FactorOptimizationState:
        logger.info("开始因子优化工作流")
        new_state = state.copy()
        new_state["current_batch"] = 0
        new_state["total_factors_needed"] = self.total_factors_needed
        new_state["generated_factors"] = []
        new_state["evaluated_factors"] = []
        new_state["factor_table"] = pd.DataFrame(columns=["名称", "表达式", "IC综合"])
        new_state["status"] = "running"
        new_state["error"] = None
        new_state["final_result"] = None
        return new_state

    def _generate_factors_node(self, state: FactorOptimizationState) -> FactorOptimizationState:
        try:
            logger.info(f"开始生成第 {state['current_batch'] + 1} 批因子")
            system_prompt, human_prompt = self._create_factor_generation_prompt(state)
            messages = [SystemMessage(content=system_prompt)]
            if human_prompt:
                messages.append(HumanMessage(content=human_prompt))
            response = self.llm.invoke(messages)
            factors_dict = self._parse_factor_response(response.content)
            new_state = state.copy()
            new_state["generated_factors"] = factors_dict
            new_state["current_batch"] = state["current_batch"] + 1
            new_state["status"] = "factors_generated"
            logger.info(f"成功生成 {len(factors_dict)} 个因子")
            return new_state
        except Exception as e:
            logger.error(f"生成因子失败: {e}")
            new_state = state.copy()
            new_state["status"] = "error"
            new_state["error"] = str(e)
            return new_state

    def _evaluate_factors_node(self, state: FactorOptimizationState) -> FactorOptimizationState:
        try:
            logger.info("开始评价因子")
            factors_dict = state["generated_factors"]
            evaluated_results = []
            df_factors = compute_factors_for_pool(
                pro = self.pro,
                factor_dict=factors_dict,
                stock_pool=self.stock_pool,
                start_date=self.start_date,
                end_date=self.end_date
            )
            for factor_name, factor_expr in factors_dict.items():
                factor_series = df_factors[factor_name]
                try:
                    eval_result = self._evaluate_factor(factor_name, factor_expr, factor_series)
                    if eval_result is not None:
                        evaluated_results.append(eval_result)
                except Exception as e:
                    logger.error(f"评价因子 {factor_name} 时出错: {e}")
            new_state = state.copy()
            new_state["evaluated_factors"] = evaluated_results
            new_state["status"] = "factors_evaluated"
            logger.info(f"成功评价 {len(evaluated_results)} 个因子")
            return new_state
        except Exception as e:
            logger.error(f"评价因子失败: {e}")
            new_state = state.copy()
            new_state["status"] = "error"
            new_state["error"] = str(e)
            return new_state

    def _collect_results_node(self, state: FactorOptimizationState) -> FactorOptimizationState:
        try:
            logger.info("收集因子评价结果")
            evaluated_factors = state["evaluated_factors"]
            current_table = state["factor_table"]
            new_rows = pd.DataFrame(evaluated_factors)
            updated_table = pd.concat([current_table, new_rows], ignore_index=True)
            new_state = state.copy()
            new_state["factor_table"] = updated_table
            new_state["status"] = "results_collected"
            total_generated = len(updated_table)
            logger.info(f"当前已收集 {total_generated} 个因子，目标 {state['total_factors_needed']}")
            return new_state
        except Exception as e:
            logger.error(f"收集结果失败: {e}")
            new_state = state.copy()
            new_state["status"] = "error"
            new_state["error"] = str(e)
            return new_state

    def _end_node(self, state: FactorOptimizationState) -> FactorOptimizationState:
        try:
            logger.info("因子优化工作流结束")
            factor_table = state["factor_table"]
            if factor_table.empty:
                final_result = {"error": "没有生成任何有效因子"}
            else:
                sorted_factors = factor_table.sort_values("IC综合", ascending=False).head(self.top_n)
                final_result = {
                    "total_factors_generated": len(factor_table),
                    f"top_{self.top_n}_factors": sorted_factors.to_dict("records"),
                    "summary_stats": {
                        "mean_ic_composite": sorted_factors["IC综合"].mean(),
                        "std_ic_composite": sorted_factors["IC综合"].std(),
                        "max_ic_composite": sorted_factors["IC综合"].max(),
                        "min_ic_composite": sorted_factors["IC综合"].min(),
                        "positive_ic_ratio": (sorted_factors["IC综合"] > 0).mean(),
                    },
                }
            new_state = state.copy()
            new_state["status"] = "completed"
            new_state["final_result"] = final_result
            logger.info(f"工作流完成，返回前 {min(self.top_n, len(factor_table))} 个因子")
            return new_state
        except Exception as e:
            logger.error(f"结束节点处理失败: {e}")
            new_state = state.copy()
            new_state["status"] = "error"
            new_state["error"] = str(e)
            return new_state

    def _should_continue(self, state: FactorOptimizationState) -> str:
        total_generated = len(state["factor_table"])
        if total_generated >= state["total_factors_needed"]:
            return "end"
        else:
            return "continue"

    # ---------- 提示词与解析 ----------
    def _create_factor_generation_prompt(self, state: FactorOptimizationState) -> tuple[str, str]:
        # 系统提示词（根据你的算子库修改）
        system_prompt = f"""
你是一个专业的量化投资专家，需要生成 {self.batch_size} 个有金融意义的量化因子。

【日线数据字段】（使用时需加 $ 符号）：
- $open   : 开盘价
- $high   : 最高价
- $low    : 最低价
- $close  : 收盘价
- $vol    : 成交量（手）
- $amount : 成交金额（元）

【支持的算子及使用示例】：
- SMA(x, n)     : 简单移动平均 → SMA($close, 5)
- EMA(x, n)     : 指数移动平均 → EMA($close, 12)
- Delta(x, n)   : 今日减 n 日前值 → Delta($close, 5)
- Mean(x, n)    : 滚动均值 → Mean($close, 20)
- Std(x, n)     : 滚动标准差 → Std($close, 20)
- Ref(x, n)     : n 期前值 → Ref($close, 1)
- Corr(x, y, n) : 滚动相关系数 → Corr($close, $vol, 20)
- Cov(x, y, n)  : 滚动协方差 → Cov($close, $vol, 20)
- Scale(x,a)      : 线性缩放（除以绝对值之和）→ Scale($close, 0.1)
- Clip(x, l, u) : 截断到 [l, u] → Clip($close, -0.1, 0.1)
- Abs(x)        : 绝对值 → Abs($close - $open)
- Log(x)        : 自然对数 → Log(1 + $vol)
- Sign(x)       : 符号函数 → Sign($close - Ref($close, 1))
- Sqrt(x)       : 平方根 → Sqrt($vol)
- Rank(x, n)    : 时间序列排名（0~1）→ Rank($close, 20)
- RSI(x, n)     : 相对强弱指数 → RSI($close, 14)

【要求】：
1. 每个因子必须有实际的金融意义，能够反映股票的某种特征。
2. 因子表达式必须使用上述支持的算子。
3. 输出格式必须是 JSON 字典：{{"因子名": "表达式"}}。
4. 禁止使用未来数据（如 Ref($close, -1) 是不允许的）。
5. 避免产生 NaN：分母加 1e-12，对数用 Log(1 + x) 等。

【示例因子】：
{{
    "momentum": "Mean($close, 20) / (Mean($close, 60) + 1e-12) - 1",
    "volume_contrarian": "-1 * Corr($vol, ($close - Ref($close, 1))/Ref($close, 1), 20)",
    "volatility_ratio": "Std($close, 10) / (Std($close, 30) + 1e-12)",
    "log_volume": "Log(1 + $vol)",
    "reversal": "($close - Ref($close, 5)) / (Ref($close, 5) + 1e-12)"
}}

请生成 {self.batch_size} 个创新的、有金融意义的不重复因子，返回 JSON 字典。
"""
        human_prompt = ""
        existing_factors = state["factor_table"]
        if not existing_factors.empty:
            human_prompt = "已有的因子及其 IC 综合值（用于参考，避免重复）：\n"
            for _, row in existing_factors.iterrows():
                human_prompt += f"- {row['名称']}: {row['表达式']} (IC综合: {row['IC综合']:.4f})(平均IC: {row['mean_IC']:.4f})(IC标准差: {row['std_IC']:.4f})(平均rankIC: {row['mean_RankIC']:.4f})(RankIC标准差: {row['std_RankIC']:.4f})\n"
            human_prompt += "\n请基于这些结果生成更好的因子，避免重复，并尝试改进高 IC 综合因子的变体或组合。"
        return system_prompt, human_prompt

    def _parse_factor_response(self, response: str) -> Dict[str, str]:
        try:
            factors_dict = json.loads(response)
            return factors_dict
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            logger.warning("无法解析LLM响应，返回空字典")
            return {}

    # ---------- 运行入口 ----------
    def run_optimization(self) -> Dict[str, Any]:
        app = self.workflow.compile()
        initial_state = FactorOptimizationState(
            current_batch=0,
            total_factors_needed=self.total_factors_needed,
            generated_factors=[],
            evaluated_factors=[],
            factor_table=pd.DataFrame(columns=["名称", "表达式", "IC综合"]),
            status="initialized",
            error=None,
            final_result=None,
        )
        final_state = app.invoke(initial_state)
        if final_state["status"] == "completed":
            return final_state["final_result"]
        else:
            return {"status": "error", "error": final_state.get("error")}


# ---------- 便捷函数 ----------
def optimize_factors_with_llm(
    pro,
    stock_pool: List[str],
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
    llm_model_uri: str = "https://api.deepseek.com",
    llm_model: str = "deepseek-chat",
    batch_size: int = 20,
    total_factors_needed: int = 100,
    top_n: int = 20,
    trim_quantile: tuple = (0.01, 0.99),
    min_samples: int = 5,
) -> Dict[str, Any]:
    optimizer = FactorOptimizer(
        pro=pro,
        stock_pool=stock_pool,
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        llm_model_uri=llm_model_uri,
        llm_model=llm_model,
        batch_size=batch_size,
        total_factors_needed=total_factors_needed,
        top_n=top_n,
        trim_quantile=trim_quantile,
        min_samples=min_samples,
    )
    return optimizer.run_optimization()

def get_api_key_from_env(api_name: str = "DEEPSEEK_API_KEY") -> Optional[str]:
    load_dotenv('config/llm.env')
    api = os.getenv(api_name)
    return api