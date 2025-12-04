"""Bitkub exchange adapter for Freqtrade (dry-run only)

This module provides a Bitkub class that integrates with Freqtrade
for dry-run/paper trading using the Bitkub exchange. This implementation
uses the bitkub-python library for fetching market data.

Limitations:
- Dry-run mode only (no real money trading)
- Spot trading only (no futures, margin, or leverage)
- No WebSocket support
- Only basic exchange functions for dry-run simulation
"""

import logging
from datetime import UTC, datetime
from typing import Any

from freqtrade.constants import Config, ExchangeConfig
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Bitkub(Exchange):
    """Bitkub exchange class for Freqtrade (dry-run mode only).

    This exchange adapter uses the bitkub-python library to fetch market data
    from Bitkub exchange. It only supports dry-run mode for paper trading
    simulation. Real money trading is not supported.

    Attributes:
        _ft_has: Exchange capabilities configuration.
    """

    _ft_has: FtHas = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["GTC"],
        "ohlcv_has_history": False,
        "trades_has_history": False,
        "ws_enabled": False,
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
    ]

    def __init__(
        self,
        config: Config,
        *,
        exchange_config: ExchangeConfig | None = None,
        validate: bool = True,
        load_leverage_tiers: bool = False,
    ) -> None:
        """Initialize Bitkub exchange adapter.

        Args:
            config: Freqtrade configuration dictionary.
            exchange_config: Optional exchange-specific configuration.
            validate: Whether to validate the configuration.
            load_leverage_tiers: Whether to load leverage tiers (ignored for Bitkub).

        Raises:
            OperationalException: If dry_run is not enabled.
        """
        # Initialize attributes that __del__ might access before parent __init__
        self._exchange_ws = None
        self._api_async = None
        self._ws_async = None
        self.loop = None

        # Bitkub only supports dry-run mode
        if not config.get("dry_run", False):
            raise OperationalException(
                "Bitkub exchange is only supported in dry-run mode. "
                "Please set 'dry_run': true in your configuration."
            )

        # Initialize bitkub-python client for market data
        self._bitkub_client = None
        try:
            from bitkub import Client

            self._bitkub_client = Client()
            logger.info("Bitkub client initialized successfully for dry-run mode.")
        except ImportError:
            logger.warning(
                "bitkub-python library not found. Install it with: pip install bitkub-python"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Bitkub client: {e}")

        # Call parent constructor - this will use ccxt fallback
        # We override _init_ccxt to handle bitkub not being in ccxt
        super().__init__(
            config,
            exchange_config=exchange_config,
            validate=validate,
            load_leverage_tiers=False,  # Bitkub doesn't support futures
        )

    def _init_ccxt(
        self, exchange_config: dict[str, Any], sync: bool, ccxt_kwargs: dict[str, Any]
    ) -> Any:
        """Initialize a mock CCXT-like object for Bitkub.

        Since Bitkub is not supported by CCXT, we create a mock object that
        provides the minimal interface required by Freqtrade's dry-run mode.

        Args:
            exchange_config: Exchange configuration dictionary.
            sync: Whether to use sync or async mode.
            ccxt_kwargs: Additional CCXT configuration.

        Returns:
            A mock CCXT-like object with minimal Bitkub functionality.
        """
        return _BitkubMockCcxt(self._bitkub_client)

    @property
    def name(self) -> str:
        """Return exchange name."""
        return "Bitkub"

    @property
    def id(self) -> str:
        """Return exchange ID."""
        return "bitkub"


class _BitkubMockCcxt:
    """Mock CCXT-like object for Bitkub exchange.

    This class provides the minimal interface that Freqtrade expects from
    a CCXT exchange object, using bitkub-python for actual API calls when
    available.
    """

    def __init__(self, bitkub_client: Any = None) -> None:
        """Initialize mock CCXT object.

        Args:
            bitkub_client: Optional bitkub-python Client instance.
        """
        self._client = bitkub_client
        self.name = "Bitkub"
        self.id = "bitkub"
        self.timeframes: dict[str, str] = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }
        self.has: dict[str, bool] = {
            "fetchOHLCV": True,
            "fetchTicker": True,
            "fetchTickers": True,
            "fetchOrderBook": True,
            "fetchL2OrderBook": True,
            "fetchBalance": True,
            "createOrder": True,
            "createLimitOrder": True,
            "createMarketOrder": True,
            "cancelOrder": True,
            "fetchOrder": True,
            "fetchOpenOrders": True,
            "fetchClosedOrders": True,
            "fetchOrders": True,
            "fetchMyTrades": True,
            "fetchTrades": True,
            "fetchMarkets": True,
        }
        self.markets: dict[str, Any] = {}
        self.currencies: dict[str, Any] = {}
        self.precisionMode = 2  # DECIMAL_PLACES
        self.session = None  # No async session needed for dry-run

        # Load markets on initialization if client is available
        if self._client:
            self._load_markets()

    def _load_markets(self) -> None:
        """Load markets from Bitkub API."""
        try:
            symbols = self._client.fetch_symbols()
            if symbols and "result" in symbols:
                for symbol_data in symbols["result"]:
                    # Bitkub uses format like "THB_BTC"
                    symbol = symbol_data.get("symbol", "")
                    if "_" in symbol:
                        quote, base = symbol.split("_", 1)
                        pair = f"{base}/{quote}"
                        self.markets[pair] = {
                            "id": symbol,
                            "symbol": pair,
                            "base": base,
                            "quote": quote,
                            "active": True,
                            "spot": True,
                            "margin": False,
                            "swap": False,
                            "future": False,
                            "option": False,
                            "type": "spot",
                            "precision": {"amount": 8, "price": 2},
                            "limits": {
                                "amount": {"min": 0.00000001, "max": None},
                                "price": {"min": 0.01, "max": None},
                                "cost": {"min": 1, "max": None},
                            },
                            "info": symbol_data,
                        }
            logger.info(f"Loaded {len(self.markets)} markets from Bitkub")
        except Exception as e:
            logger.warning(f"Failed to load markets from Bitkub: {e}")
            # Provide default markets for dry-run
            self._set_default_markets()

    def _set_default_markets(self) -> None:
        """Set default markets when API is unavailable."""
        default_pairs = [
            ("BTC", "THB"),
            ("ETH", "THB"),
            ("USDT", "THB"),
            ("XRP", "THB"),
            ("ADA", "THB"),
        ]
        for base, quote in default_pairs:
            pair = f"{base}/{quote}"
            self.markets[pair] = {
                "id": f"{quote}_{base}",
                "symbol": pair,
                "base": base,
                "quote": quote,
                "active": True,
                "spot": True,
                "margin": False,
                "swap": False,
                "future": False,
                "option": False,
                "type": "spot",
                "precision": {"amount": 8, "price": 2},
                "limits": {
                    "amount": {"min": 0.00000001, "max": None},
                    "price": {"min": 0.01, "max": None},
                    "cost": {"min": 1, "max": None},
                },
                "info": {},
            }

    def load_markets(self, reload: bool = False) -> dict[str, Any]:
        """Load or reload markets.

        Args:
            reload: Whether to force reload markets.

        Returns:
            Dictionary of markets.
        """
        if reload or not self.markets:
            self._load_markets()
        return self.markets

    def fetch_ticker(self, symbol: str, params: dict | None = None) -> dict[str, Any]:
        """Fetch ticker for a symbol.

        Args:
            symbol: Trading pair symbol.
            params: Additional parameters.

        Returns:
            Ticker data dictionary.
        """
        if self._client:
            try:
                tickers = self._client.fetch_tickers()
                if tickers:
                    # Convert symbol format: "BTC/THB" -> "THB_BTC"
                    base, quote = symbol.split("/")
                    bitkub_symbol = f"{quote}_{base}"
                    if bitkub_symbol in tickers:
                        t = tickers[bitkub_symbol]
                        return {
                            "symbol": symbol,
                            "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                            "datetime": datetime.now(UTC).isoformat(),
                            "high": float(t.get("high24hr", 0)),
                            "low": float(t.get("low24hr", 0)),
                            "bid": float(t.get("highestBid", 0)),
                            "ask": float(t.get("lowestAsk", 0)),
                            "last": float(t.get("last", 0)),
                            "close": float(t.get("last", 0)),
                            "baseVolume": float(t.get("baseVolume", 0)),
                            "quoteVolume": float(t.get("quoteVolume", 0)),
                            "info": t,
                        }
            except Exception as e:
                logger.warning(f"Failed to fetch ticker for {symbol}: {e}")

        # Return mock ticker for dry-run
        return self._mock_ticker(symbol)

    def _mock_ticker(self, symbol: str) -> dict[str, Any]:
        """Generate mock ticker data for dry-run.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Mock ticker data dictionary.
        """
        now = datetime.now(UTC)
        return {
            "symbol": symbol,
            "timestamp": int(now.timestamp() * 1000),
            "datetime": now.isoformat(),
            "high": 0,
            "low": 0,
            "bid": 0,
            "ask": 0,
            "last": 0,
            "close": 0,
            "baseVolume": 0,
            "quoteVolume": 0,
            "info": {},
        }

    def fetch_tickers(
        self, symbols: list[str] | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        """Fetch tickers for multiple symbols.

        Args:
            symbols: List of trading pair symbols.
            params: Additional parameters.

        Returns:
            Dictionary of tickers keyed by symbol.
        """
        if symbols is None:
            symbols = list(self.markets.keys())

        result = {}
        for symbol in symbols:
            result[symbol] = self.fetch_ticker(symbol)
        return result

    def fetch_order_book(
        self, symbol: str, limit: int | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        """Fetch order book for a symbol.

        Args:
            symbol: Trading pair symbol.
            limit: Maximum number of orders to return.
            params: Additional parameters.

        Returns:
            Order book dictionary with bids and asks.
        """
        if self._client:
            try:
                base, quote = symbol.split("/")
                bitkub_symbol = f"{quote}_{base}"
                bids = self._client.fetch_bids(sym=bitkub_symbol, lmt=limit or 10)
                asks = self._client.fetch_asks(sym=bitkub_symbol, lmt=limit or 10)

                return {
                    "symbol": symbol,
                    "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                    "datetime": datetime.now(UTC).isoformat(),
                    "bids": [
                        [float(b.get("price", 0)), float(b.get("amount", 0))]
                        for b in (bids.get("result", []) if bids else [])
                    ],
                    "asks": [
                        [float(a.get("price", 0)), float(a.get("amount", 0))]
                        for a in (asks.get("result", []) if asks else [])
                    ],
                }
            except Exception as e:
                logger.warning(f"Failed to fetch order book for {symbol}: {e}")

        # Return mock order book for dry-run
        return {
            "symbol": symbol,
            "timestamp": int(datetime.now(UTC).timestamp() * 1000),
            "datetime": datetime.now(UTC).isoformat(),
            "bids": [],
            "asks": [],
        }

    def fetch_l2_order_book(
        self, symbol: str, limit: int | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        """Fetch L2 order book (alias for fetch_order_book).

        Args:
            symbol: Trading pair symbol.
            limit: Maximum number of orders to return.
            params: Additional parameters.

        Returns:
            Order book dictionary with bids and asks.
        """
        return self.fetch_order_book(symbol, limit, params)

    def fetch_balance(self, params: dict | None = None) -> dict[str, Any]:
        """Fetch account balance.

        For dry-run mode, returns empty balance as actual trading is not supported.

        Args:
            params: Additional parameters.

        Returns:
            Balance dictionary (empty for dry-run).
        """
        return {"info": {}, "free": {}, "used": {}, "total": {}}

    async def close(self) -> None:
        """Close async session (no-op for mock)."""
        pass
