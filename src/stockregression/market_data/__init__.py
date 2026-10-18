"""Year-partitioned Parquet market data with transactional version manifests.

The public entry point is :class:`MarketDataService`. SQLite keeps catalogue,
version and quality metadata; DuckDB reads immutable Parquet bar partitions.
"""

from .service import MarketDataService

__all__ = ["MarketDataService"]
