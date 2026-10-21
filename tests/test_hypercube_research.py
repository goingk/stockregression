"""稳健 J 搜索格点的共享内核契约。

搜索格点是 v5 稳健 J 研究与超正方体邻域共用的内核：研究细粒度只决定采样
步长，永远不改变参数可以取的最小单位。旧版的 v3 超正方体选择器与 v4 连通域
几何已随协议移除，相关用例不再保留。
"""

from datetime import date

from stockregression.models import BacktestConfig, OptimizationConfig
from stockregression.research_v4.space import ParameterSpace


def _v5_config(granularity: dict[str, float]) -> OptimizationConfig:
    base = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date=date(2024, 1, 2),
        frequency="1d", initial_shares=1000, min_shares=0, max_shares=None,
        buy_shares=100, sell_shares=100, grid_mode="geometric",
        buy_step=.0003, sell_step=.0003,
    )
    return OptimizationConfig(
        base=base,
        ranges={
            "buy_step": {"low": .003, "high": .005},
            "sell_step": {"low": .003, "high": .003},
            "buy_shares": {"low": 100, "high": 1000},
            "sell_shares": {"low": 100, "high": 1000},
        },
        strategy={"granularity": granularity},
    )


def test_research_granularity_is_independent_of_minimum_parameter_unit():
    fine = _v5_config({"buy_step": .0005, "sell_step": .0003, "buy_shares": 100, "sell_shares": 100})
    coarse = _v5_config({"buy_step": .001, "sell_step": .0003, "buy_shares": 500, "sell_shares": 100})

    fine_space, coarse_space = ParameterSpace(fine), ParameterSpace(coarse)

    assert fine_space.value("buy_step", 6) == .003
    assert fine_space.dimensions["buy_step"].low == 6
    assert fine_space.dimensions["buy_step"].count > coarse_space.dimensions["buy_step"].count
    assert fine_space.dimensions["buy_shares"].count > coarse_space.dimensions["buy_shares"].count
    # 最小单位恒为 0.0001 与 100 股，变化的只是采样步长。
    assert fine_space.value("buy_step", 1) == .0005
    assert fine_space.value("buy_shares", 1) == 100


def test_v5_search_space_exposes_only_the_four_robust_axes():
    space = ParameterSpace(_v5_config(
        {"buy_step": .0005, "sell_step": .0003, "buy_shares": 100, "sell_shares": 100},
    ))

    assert space.fields == ("buy_step", "sell_step", "buy_shares", "sell_shares")
    assert "min_shares" not in space.dimensions
    assert "max_shares" not in space.dimensions
    assert space.max_unlimited is True
