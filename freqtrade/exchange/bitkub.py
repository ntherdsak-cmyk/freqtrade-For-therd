"""Bitkub exchange adapter for Freqtrade

This module implements a custom exchange adapter for Bitkub (Thailand's leading
cryptocurrency exchange). Since Bitkub is not supported by ccxt, this adapter
directly calls the Bitkub REST API.

Documentation: https://api.bitkub.com/docs
GitHub API Docs: https://github.com/bitkub/bitkub-official-api-docs

Features:
- Full Spot trading support
- HMAC SHA256 signed requests
- Dry-run mode with mock data
- fetch_markets, fetch_balance, fetch_ohlcv, create_order, cancel_order,
  fetch_order, fetch_open_orders

Usage:
Configure in config.json:
{
    "exchange": {
        "name": "bitkub",
        "key": "YOUR_API_KEY",
        "secret": "YOUR_API_SECRET"
    }
}
"""

import hashlib
import hmac
import logging
import time
from datetime import datetime
from typing import Any

import requests

from freqtrade.constants import BuySell, Config
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import (
    DDosProtection,
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderException,
    OperationalException,
    TemporaryError,
)
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import CcxtBalances, CcxtOrder, FtHas


logger = logging.getLogger(__name__)


class Bitkub(Exchange):
    """Bitkub exchange class.

    Contains adjustments needed for Freqtrade to work with Bitkub exchange.
    Bitkub is not supported by ccxt, so this class implements direct REST API calls.

    Note: Bitkub only supports Spot trading.
    """

    # Base URL for Bitkub API
    _base_url = "https://api.bitkub.com"

    # Bitkub-specific configuration
    _ft_has: FtHas = {
        "stoploss_on_exchange": False,  # Bitkub doesn't support stoploss orders
        "order_time_in_force": ["GTC"],
        "ohlcv_has_history": True,
        "ohlcv_candle_limit": 1000,  # Max candles per request
        "trades_has_history": False,
        "l2_limit_range": None,
        "tickers_have_quoteVolume": True,
        "tickers_have_percentage": True,
        "tickers_have_bid_ask": True,
        "tickers_have_price": True,
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
    ]

    def __init__(self, config: Config, *args, **kwargs) -> None:
        """Initialize Bitkub exchange.

        Args:
            config: Freqtrade configuration
        """
        # Extract API credentials before parent init
        exchange_config = config.get("exchange", {})
        self._api_key = exchange_config.get("key", "")
        self._api_secret = exchange_config.get("secret", "")

        # For Bitkub, we need to handle the fact that it's not in ccxt
        # We'll set a flag to use our custom implementation
        self._use_bitkub_api = True

        # Initialize session for API calls
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

        # Cache for markets data
        self._bitkub_markets: dict = {}
        self._bitkub_symbols: dict = {}

        # Skip parent ccxt initialization for non-dry-run mode
        # Instead, initialize only what we need
        if not config.get("dry_run", False):
            self._init_bitkub_exchange(config)
        else:
            # For dry-run, we can use a mock ccxt exchange
            super().__init__(config, *args, **kwargs)

    def _init_bitkub_exchange(self, config: Config) -> None:
        """Initialize Bitkub-specific exchange settings."""
        import asyncio
        from copy import deepcopy

        from freqtrade.misc import deep_merge_dicts

        self._config = config
        self.trading_mode = TradingMode.SPOT
        self.margin_mode = MarginMode.NONE
        self._markets: dict = {}
        self._trading_fees: dict = {}
        self._leverage_tiers: dict = {}
        self._dry_run_open_orders: dict = {}
        self._klines: dict = {}
        self._trades: dict = {}
        self._pairs_last_refresh_time: dict = {}
        self._last_markets_refresh: int = 0
        self._exchange_ws = None  # Not used for Bitkub

        # Initialize async loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Build ft_has
        self._ft_has = deep_merge_dicts(self._ft_has, deepcopy(self._ft_has_default))

        # Load markets
        self._load_bitkub_markets()

    def close(self) -> None:
        """Clean up resources.

        Override parent method to handle Bitkub-specific cleanup.
        """
        # Check if _exchange_ws exists before calling parent close
        if hasattr(self, "_exchange_ws") and self._exchange_ws:
            self._exchange_ws.cleanup()

        # Close the requests session
        if hasattr(self, "_session"):
            self._session.close()

        # Close async loop if it exists
        if hasattr(self, "loop") and self.loop and not self.loop.is_closed():
            self.loop.close()

    def _get_server_time(self) -> int:
        """Get server time from Bitkub API.

        Returns:
            Server timestamp in milliseconds
        """
        try:
            response = self._session.get(f"{self._base_url}/api/v3/servertime")
            response.raise_for_status()
            return int(response.text)
        except requests.RequestException as e:
            logger.warning(f"Failed to get server time: {e}")
            return int(time.time() * 1000)

    def _generate_signature(self, timestamp: int, payload: str = "") -> str:
        """Generate HMAC SHA256 signature for authenticated requests.

        Args:
            timestamp: Current timestamp in milliseconds
            payload: Request payload as JSON string

        Returns:
            HMAC SHA256 signature
        """
        message = f"{timestamp}{payload}"
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _make_public_request(self, endpoint: str, params: dict | None = None) -> Any:
        """Make a public (unsigned) API request.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            API response data

        Raises:
            TemporaryError: On temporary API errors
            OperationalException: On permanent API errors
        """
        url = f"{self._base_url}{endpoint}"
        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and data.get("error", 0) != 0:
                error_code = data.get("error", 0)
                error_msg = self._get_error_message(error_code)
                raise ExchangeError(f"Bitkub API error {error_code}: {error_msg}")

            return data
        except requests.exceptions.Timeout as e:
            raise TemporaryError(f"Bitkub API timeout: {e}") from e
        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 429:
                    raise DDosProtection(f"Rate limit exceeded: {e}") from e
            raise TemporaryError(f"Bitkub API request failed: {e}") from e

    def _make_private_request(
        self,
        endpoint: str,
        method: str = "POST",
        payload: dict | None = None
    ) -> Any:
        """Make a private (signed) API request.

        Args:
            endpoint: API endpoint path
            method: HTTP method (POST, GET, etc.)
            payload: Request payload

        Returns:
            API response data

        Raises:
            TemporaryError: On temporary API errors
            OperationalException: On permanent API errors
            InsufficientFundsError: When balance is insufficient
        """
        if not self._api_key or not self._api_secret:
            raise OperationalException(
                "API key and secret are required for private endpoints"
            )

        url = f"{self._base_url}{endpoint}"
        timestamp = self._get_server_time()

        if payload is None:
            payload = {}

        # Add timestamp to payload
        payload["ts"] = timestamp

        import json
        payload_str = json.dumps(payload, separators=(",", ":"))
        signature = self._generate_signature(timestamp, payload_str)

        headers = {
            "X-BTK-APIKEY": self._api_key,
            "X-BTK-TIMESTAMP": str(timestamp),
            "X-BTK-SIGN": signature,
        }

        try:
            if method.upper() == "POST":
                response = self._session.post(
                    url, json=payload, headers=headers, timeout=30
                )
            else:
                response = self._session.get(
                    url, params=payload, headers=headers, timeout=30
                )

            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and data.get("error", 0) != 0:
                error_code = data.get("error", 0)
                error_msg = self._get_error_message(error_code)

                if error_code == 3:  # Insufficient balance
                    raise InsufficientFundsError(f"Insufficient funds: {error_msg}")
                elif error_code in [1, 2]:  # Invalid JSON, missing field
                    raise InvalidOrderException(f"Invalid order: {error_msg}")
                else:
                    raise ExchangeError(f"Bitkub API error {error_code}: {error_msg}")

            return data
        except requests.exceptions.Timeout as e:
            raise TemporaryError(f"Bitkub API timeout: {e}") from e
        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 429:
                    raise DDosProtection(f"Rate limit exceeded: {e}") from e
            raise TemporaryError(f"Bitkub API request failed: {e}") from e

    def _get_error_message(self, error_code: int) -> str:
        """Get human-readable error message for Bitkub error codes.

        Args:
            error_code: Bitkub API error code

        Returns:
            Error message string
        """
        error_messages = {
            0: "No error",
            1: "Invalid JSON payload",
            2: "Missing X-BTK-APIKEY",
            3: "Invalid API key",
            4: "API pending for activation",
            5: "IP not allowed",
            6: "Missing / invalid signature",
            7: "Missing timestamp",
            8: "Invalid timestamp",
            9: "Invalid user",
            10: "Invalid parameter",
            11: "Invalid symbol",
            12: "Invalid amount",
            13: "Invalid rate",
            14: "Improper rate",
            15: "Amount too low",
            16: "Failed to get balance",
            17: "Wallet is empty",
            18: "Insufficient balance",
            19: "Failed to insert order into db",
            20: "Failed to deduct balance",
            21: "Invalid order for cancellation",
            22: "Invalid side",
            23: "Failed to update order status",
            24: "Invalid order for lookup",
            25: "KYC level 1 is required",
            30: "Limit exceed",
            40: "Pending withdrawal exists",
            41: "Invalid currency for withdrawal",
            42: "Address is not in whitelist",
            43: "Failed to deduct crypto",
            44: "Failed to create withdrawal record",
            45: "Nonce has to be numeric",
            46: "Invalid nonce",
            47: "Withdrawal limit exceed",
            48: "Invalid bank account",
            49: "Bank limit exceed",
            50: "Pending withdrawal exists",
            51: "Withdrawal is under maintenance",
            52: "Invalid permission",
            53: "Invalid internal address",
            54: "Address has been deprecated",
            55: "Cancel only mode",
            90: "Server error (try again later)",
        }
        return error_messages.get(error_code, f"Unknown error ({error_code})")

    def _load_bitkub_markets(self) -> None:
        """Load markets data from Bitkub API."""
        try:
            # Get symbols info
            symbols_data = self._make_public_request("/api/market/symbols")

            if isinstance(symbols_data, dict) and symbols_data.get("error") == 0:
                symbols_list = symbols_data.get("result", [])
            elif isinstance(symbols_data, list):
                symbols_list = symbols_data
            else:
                symbols_list = []

            markets = {}
            for symbol_info in symbols_list:
                symbol = symbol_info.get("symbol", "")
                if not symbol:
                    continue

                # Bitkub symbols are like "THB_BTC" -> "BTC/THB"
                parts = symbol.split("_")
                if len(parts) != 2:
                    continue

                quote, base = parts[0], parts[1]
                pair = f"{base}/{quote}"

                markets[pair] = {
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
                    "contract": False,
                    "settle": None,
                    "settleId": None,
                    "contractSize": None,
                    "linear": None,
                    "inverse": None,
                    "expiry": None,
                    "expiryDatetime": None,
                    "strike": None,
                    "optionType": None,
                    "precision": {
                        "amount": 8,
                        "price": 2,
                    },
                    "limits": {
                        "leverage": {"min": 1, "max": 1},
                        "amount": {"min": 0.0001, "max": None},
                        "price": {"min": None, "max": None},
                        "cost": {"min": 10, "max": None},  # Min 10 THB
                    },
                    "info": symbol_info,
                    "maker": 0.0025,  # 0.25% maker fee
                    "taker": 0.0025,  # 0.25% taker fee
                }

                self._bitkub_symbols[pair] = symbol

            self._markets = markets
            self._bitkub_markets = markets
            logger.info(f"Loaded {len(markets)} markets from Bitkub")

        except Exception as e:
            logger.error(f"Failed to load Bitkub markets: {e}")
            raise OperationalException(f"Failed to load Bitkub markets: {e}") from e

    @property
    def markets(self) -> dict[str, Any]:
        """Get markets data."""
        if not self._markets and not self._config.get("dry_run", False):
            self._load_bitkub_markets()
        return self._markets

    @property
    def name(self) -> str:
        """Exchange name."""
        return "Bitkub"

    @property
    def id(self) -> str:
        """Exchange id."""
        return "bitkub"

    @retrier
    def get_balances(self) -> CcxtBalances:
        """Fetch account balances.

        Returns:
            Dict of currency balances
        """
        if self._config.get("dry_run", False):
            return {}

        try:
            response = self._make_private_request("/api/v3/market/balances")
            result = response.get("result", {})

            balances: CcxtBalances = {}
            for currency, balance_info in result.items():
                if isinstance(balance_info, dict):
                    available = float(balance_info.get("available", 0))
                    reserved = float(balance_info.get("reserved", 0))
                else:
                    available = float(balance_info)
                    reserved = 0

                total = available + reserved
                if total > 0:
                    balances[currency.upper()] = {
                        "free": available,
                        "used": reserved,
                        "total": total,
                    }

            return balances

        except Exception as e:
            logger.error(f"Failed to fetch balances: {e}")
            raise

    @retrier
    def fetch_ticker(self, pair: str) -> dict:
        """Fetch ticker data for a pair.

        Args:
            pair: Trading pair (e.g., "BTC/THB")

        Returns:
            Ticker data dict
        """
        if self._config.get("dry_run", False):
            return super().fetch_ticker(pair)

        symbol = self._bitkub_symbols.get(pair)
        if not symbol:
            raise InvalidOrderException(f"Unknown symbol: {pair}")

        try:
            response = self._make_public_request("/api/market/ticker")

            if symbol in response:
                ticker_data = response[symbol]
                return {
                    "symbol": pair,
                    "timestamp": int(time.time() * 1000),
                    "datetime": datetime.utcnow().isoformat(),
                    "high": float(ticker_data.get("high24hr", 0)),
                    "low": float(ticker_data.get("low24hr", 0)),
                    "bid": float(ticker_data.get("highestBid", 0)),
                    "bidVolume": None,
                    "ask": float(ticker_data.get("lowestAsk", 0)),
                    "askVolume": None,
                    "vwap": None,
                    "open": float(ticker_data.get("open", 0)),
                    "close": float(ticker_data.get("last", 0)),
                    "last": float(ticker_data.get("last", 0)),
                    "previousClose": None,
                    "change": float(ticker_data.get("change", 0)),
                    "percentage": float(ticker_data.get("percentChange", 0)),
                    "average": None,
                    "baseVolume": float(ticker_data.get("baseVolume", 0)),
                    "quoteVolume": float(ticker_data.get("quoteVolume", 0)),
                    "info": ticker_data,
                }
            else:
                raise ExchangeError(f"Ticker not found for {pair}")

        except Exception as e:
            logger.error(f"Failed to fetch ticker for {pair}: {e}")
            raise

    def create_order(
        self,
        *,
        pair: str,
        ordertype: str,
        side: BuySell,
        amount: float,
        rate: float,
        leverage: float = 1.0,
        reduceOnly: bool = False,
        time_in_force: str = "GTC",
    ) -> CcxtOrder:
        """Create a new order.

        Args:
            pair: Trading pair
            ordertype: Order type (limit/market)
            side: Buy or sell
            amount: Order amount
            rate: Order price
            leverage: Leverage (not supported, always 1)
            reduceOnly: Reduce only flag (not supported)
            time_in_force: Time in force (only GTC supported)

        Returns:
            Order data dict
        """
        if self._config.get("dry_run", False):
            return self.create_dry_run_order(
                pair, ordertype, side, amount,
                self.price_to_precision(pair, rate), leverage
            )

        symbol = self._bitkub_symbols.get(pair)
        if not symbol:
            raise InvalidOrderException(f"Unknown symbol: {pair}")

        try:
            # Bitkub API endpoints
            if side == "buy":
                endpoint = "/api/v3/market/place-bid"
            else:
                endpoint = "/api/v3/market/place-ask"

            payload = {
                "sym": symbol,
                "amt": amount if ordertype == "market" else amount * rate,
                "rat": rate if ordertype == "limit" else 0,
                "typ": "limit" if ordertype == "limit" else "market",
            }

            response = self._make_private_request(endpoint, payload=payload)
            result = response.get("result", {})

            order_id = str(result.get("id", ""))
            if not order_id:
                raise ExchangeError("Failed to create order: no order ID returned")

            now = datetime.utcnow()
            order: CcxtOrder = {
                "id": order_id,
                "clientOrderId": None,
                "timestamp": int(now.timestamp() * 1000),
                "datetime": now.isoformat(),
                "lastTradeTimestamp": None,
                "status": "open",
                "symbol": pair,
                "type": ordertype,
                "side": side,
                "price": rate,
                "amount": amount,
                "filled": 0,
                "remaining": amount,
                "cost": 0,
                "trades": None,
                "fee": None,
                "info": result,
                "average": None,
            }

            logger.info(f"Created {side} {ordertype} order for {pair}: {order_id}")
            return order

        except InsufficientFundsError:
            raise
        except InvalidOrderException:
            raise
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise TemporaryError(f"Failed to create order: {e}") from e

    @retrier
    def cancel_order(self, order_id: str, pair: str, params: dict | None = None) -> dict:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel
            pair: Trading pair
            params: Additional parameters

        Returns:
            Cancelled order data
        """
        if self._config.get("dry_run", False):
            return super().cancel_order(order_id, pair, params)

        symbol = self._bitkub_symbols.get(pair)
        if not symbol:
            raise InvalidOrderException(f"Unknown symbol: {pair}")

        try:
            # Determine side from order info if available
            # Bitkub requires different endpoints for bid/ask cancellation
            side = params.get("side", "sell") if params else "sell"

            if side == "buy":
                endpoint = "/api/v3/market/cancel-order"
            else:
                endpoint = "/api/v3/market/cancel-order"

            payload = {
                "sym": symbol,
                "id": order_id,
                "sd": "buy" if side == "buy" else "sell",
            }

            response = self._make_private_request(endpoint, payload=payload)

            return {
                "id": order_id,
                "symbol": pair,
                "status": "canceled",
                "info": response,
            }

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise

    @retrier
    def fetch_order(
        self, order_id: str, pair: str, params: dict | None = None
    ) -> CcxtOrder:
        """Fetch order details.

        Args:
            order_id: Order ID
            pair: Trading pair
            params: Additional parameters

        Returns:
            Order data dict
        """
        if self._config.get("dry_run", False):
            return self.fetch_dry_run_order(order_id)

        symbol = self._bitkub_symbols.get(pair)
        if not symbol:
            raise InvalidOrderException(f"Unknown symbol: {pair}")

        try:
            # Bitkub uses different endpoints for order info
            # Try to get from open orders first
            open_orders = self._fetch_bitkub_open_orders(symbol)

            for order in open_orders:
                if str(order.get("id")) == order_id:
                    return self._parse_bitkub_order(order, pair)

            # If not found in open orders, try order history
            history = self._fetch_bitkub_order_history(symbol)

            for order in history:
                if str(order.get("id")) == order_id:
                    return self._parse_bitkub_order(order, pair)

            raise InvalidOrderException(f"Order not found: {order_id}")

        except Exception as e:
            logger.error(f"Failed to fetch order {order_id}: {e}")
            raise

    def _fetch_bitkub_open_orders(self, symbol: str) -> list:
        """Fetch open orders from Bitkub API.

        Args:
            symbol: Bitkub symbol (e.g., "THB_BTC")

        Returns:
            List of open orders
        """
        payload = {"sym": symbol}
        response = self._make_private_request(
            "/api/v3/market/my-open-orders",
            payload=payload
        )
        return response.get("result", [])

    def _fetch_bitkub_order_history(self, symbol: str) -> list:
        """Fetch order history from Bitkub API.

        Args:
            symbol: Bitkub symbol

        Returns:
            List of historical orders
        """
        payload = {"sym": symbol}
        response = self._make_private_request(
            "/api/v3/market/my-order-history",
            payload=payload
        )
        return response.get("result", [])

    def _parse_bitkub_order(self, order_data: dict, pair: str) -> CcxtOrder:
        """Parse Bitkub order data to ccxt format.

        Args:
            order_data: Raw order data from Bitkub
            pair: Trading pair

        Returns:
            CcxtOrder dict
        """
        order_id = str(order_data.get("id", ""))
        side = order_data.get("side", "").lower()
        order_type = order_data.get("type", "limit").lower()
        rate = float(order_data.get("rate", 0))
        amount = float(order_data.get("amount", 0))
        filled = float(order_data.get("filled", 0))
        remaining = amount - filled

        # Determine status
        if remaining == 0 and filled > 0:
            status = "closed"
        elif order_data.get("cancelled", False):
            status = "canceled"
        else:
            status = "open"

        timestamp = order_data.get("ts", int(time.time() * 1000))
        if isinstance(timestamp, (int, float)) and timestamp < 10000000000:
            timestamp = int(timestamp * 1000)

        return {
            "id": order_id,
            "clientOrderId": None,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp / 1000).isoformat(),
            "lastTradeTimestamp": None,
            "status": status,
            "symbol": pair,
            "type": order_type,
            "side": side,
            "price": rate,
            "amount": amount,
            "filled": filled,
            "remaining": remaining,
            "cost": filled * rate,
            "trades": None,
            "fee": None,
            "info": order_data,
            "average": rate if filled > 0 else None,
        }

    @retrier
    def fetch_open_orders(self, pair: str = "", params: dict | None = None) -> list:
        """Fetch all open orders.

        Args:
            pair: Trading pair (optional, fetches all if empty)
            params: Additional parameters

        Returns:
            List of open orders
        """
        if self._config.get("dry_run", False):
            return list(self._dry_run_open_orders.values())

        orders = []

        if pair:
            symbol = self._bitkub_symbols.get(pair)
            if symbol:
                raw_orders = self._fetch_bitkub_open_orders(symbol)
                for order in raw_orders:
                    orders.append(self._parse_bitkub_order(order, pair))
        else:
            # Fetch orders for all pairs
            for p, symbol in self._bitkub_symbols.items():
                try:
                    raw_orders = self._fetch_bitkub_open_orders(symbol)
                    for order in raw_orders:
                        orders.append(self._parse_bitkub_order(order, p))
                except Exception as e:
                    logger.warning(f"Failed to fetch orders for {p}: {e}")

        return orders

    def fetch_l2_order_book(self, pair: str, limit: int = 20) -> dict:
        """Fetch order book for a pair.

        Args:
            pair: Trading pair
            limit: Number of levels to fetch

        Returns:
            Order book dict with bids and asks
        """
        if self._config.get("dry_run", False):
            return {"bids": [], "asks": [], "timestamp": None, "datetime": None}

        symbol = self._bitkub_symbols.get(pair)
        if not symbol:
            raise InvalidOrderException(f"Unknown symbol: {pair}")

        try:
            response = self._make_public_request(
                "/api/market/books",
                params={"sym": symbol, "lmt": limit}
            )

            bids = []
            asks = []

            if isinstance(response, dict):
                bids_data = response.get("bids", [])
                asks_data = response.get("asks", [])

                for bid in bids_data[:limit]:
                    if isinstance(bid, list) and len(bid) >= 2:
                        bids.append([float(bid[0]), float(bid[1])])

                for ask in asks_data[:limit]:
                    if isinstance(ask, list) and len(ask) >= 2:
                        asks.append([float(ask[0]), float(ask[1])])

            return {
                "bids": bids,
                "asks": asks,
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to fetch order book for {pair}: {e}")
            raise

    def get_historic_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        candle_type: Any,
        is_new_pair: bool = False,
        until_ms: int | None = None,
    ) -> Any:
        """Fetch historical OHLCV data.

        Args:
            pair: Trading pair
            timeframe: Candle timeframe
            since_ms: Start timestamp in milliseconds
            candle_type: Candle type
            is_new_pair: Whether this is a new pair
            until_ms: End timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        if self._config.get("dry_run", False):
            return super().get_historic_ohlcv(
                pair, timeframe, since_ms, candle_type, is_new_pair, until_ms
            )

        symbol = self._bitkub_symbols.get(pair)
        if not symbol:
            logger.warning(f"Unknown symbol for OHLCV: {pair}")
            from pandas import DataFrame
            return DataFrame()

        # Map timeframe to Bitkub resolution
        resolution_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }

        resolution = resolution_map.get(timeframe, 3600)

        try:
            from_ts = int(since_ms / 1000)
            to_ts = int(until_ms / 1000) if until_ms else int(time.time())

            response = self._make_public_request(
                "/tradingview/history",
                params={
                    "symbol": symbol,
                    "resolution": resolution,
                    "from": from_ts,
                    "to": to_ts,
                }
            )

            # Parse TradingView format response
            if response.get("s") != "ok":
                logger.warning(f"OHLCV fetch failed: {response}")
                from pandas import DataFrame
                return DataFrame()

            timestamps = response.get("t", [])
            opens = response.get("o", [])
            highs = response.get("h", [])
            lows = response.get("l", [])
            closes = response.get("c", [])
            volumes = response.get("v", [])

            # Convert to OHLCV list format
            ohlcv_data = []
            for i in range(len(timestamps)):
                ohlcv_data.append([
                    timestamps[i] * 1000,  # timestamp in ms
                    float(opens[i]) if opens else 0,
                    float(highs[i]) if highs else 0,
                    float(lows[i]) if lows else 0,
                    float(closes[i]) if closes else 0,
                    float(volumes[i]) if volumes else 0,
                ])

            from freqtrade.data.converter import ohlcv_to_dataframe
            return ohlcv_to_dataframe(ohlcv_data, timeframe, pair=pair)

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {pair}: {e}")
            from pandas import DataFrame
            return DataFrame()

    def reload_markets(self, force: bool = False, **kwargs) -> None:
        """Reload markets data.

        Args:
            force: Force reload even if recently loaded
        """
        if self._config.get("dry_run", False):
            super().reload_markets(force, **kwargs)
        else:
            if force or not self._markets:
                self._load_bitkub_markets()

    def exchange_has(self, endpoint: str) -> bool:
        """Check if exchange supports an endpoint.

        Args:
            endpoint: Endpoint name

        Returns:
            True if supported
        """
        # Define supported endpoints
        supported = {
            "fetchOHLCV": True,
            "fetchTicker": True,
            "fetchTickers": True,
            "fetchOrderBook": True,
            "fetchL2OrderBook": True,
            "createOrder": True,
            "cancelOrder": True,
            "fetchOrder": True,
            "fetchOpenOrders": True,
            "fetchBalance": True,
            "createMarketOrder": True,
            "createLimitOrder": True,
            # Not supported
            "fetchFundingHistory": False,
            "fetchPositions": False,
            "fetchLeverageTiers": False,
            "setLeverage": False,
            "setMarginMode": False,
            "watchOHLCV": False,
        }
        return supported.get(endpoint, False)
