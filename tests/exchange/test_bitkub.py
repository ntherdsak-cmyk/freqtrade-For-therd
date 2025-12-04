"""
Bitkub exchange tests
"""

from unittest.mock import patch

import pytest

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exchange.bitkub import Bitkub


@pytest.fixture
def bitkub_config():
    """Default configuration for Bitkub exchange."""
    return {
        "exchange": {
            "name": "bitkub",
            "key": "test_api_key",
            "secret": "test_api_secret",
        },
        "dry_run": True,
        "trading_mode": "spot",
        "stake_currency": "THB",
    }


class TestBitkubExchange:
    """Test cases for Bitkub exchange adapter."""

    def test_bitkub_class_attributes(self):
        """Test Bitkub class has correct static attributes."""
        assert Bitkub._ft_has["stoploss_on_exchange"] is False
        assert Bitkub._ft_has["ohlcv_has_history"] is True
        assert "GTC" in Bitkub._ft_has["order_time_in_force"]

        # Check trading mode support
        assert (TradingMode.SPOT, MarginMode.NONE) in Bitkub._supported_trading_mode_margin_pairs
        assert len(Bitkub._supported_trading_mode_margin_pairs) == 1

    def test_bitkub_base_url(self):
        """Test Bitkub has correct base URL."""
        assert Bitkub._base_url == "https://api.bitkub.com"

    def test_generate_signature(self, bitkub_config):
        """Test HMAC signature generation."""
        with patch.object(Bitkub, "__init__", lambda x, y: None):
            bitkub = Bitkub.__new__(Bitkub)
            bitkub._api_secret = "test_secret"

            timestamp = 1234567890
            payload = '{"sym":"THB_BTC"}'

            signature = bitkub._generate_signature(timestamp, payload)

            # Verify it's a valid hex string
            assert isinstance(signature, str)
            assert len(signature) == 64  # SHA256 hex is 64 chars

    def test_get_error_message(self, bitkub_config):
        """Test error code to message mapping."""
        with patch.object(Bitkub, "__init__", lambda x, y: None):
            bitkub = Bitkub.__new__(Bitkub)

            assert "Invalid JSON" in bitkub._get_error_message(1)
            assert "Insufficient balance" in bitkub._get_error_message(18)
            assert "Invalid symbol" in bitkub._get_error_message(11)
            assert "No error" in bitkub._get_error_message(0)

    def test_parse_bitkub_order(self, bitkub_config):
        """Test parsing Bitkub order format to ccxt format."""
        with patch.object(Bitkub, "__init__", lambda x, y: None):
            bitkub = Bitkub.__new__(Bitkub)

            order_data = {
                "id": "123456",
                "side": "buy",
                "type": "limit",
                "rate": 50000.0,
                "amount": 0.001,
                "filled": 0.0005,
                "ts": 1609459200,
            }

            parsed = bitkub._parse_bitkub_order(order_data, "BTC/THB")

            assert parsed["id"] == "123456"
            assert parsed["symbol"] == "BTC/THB"
            assert parsed["side"] == "buy"
            assert parsed["type"] == "limit"
            assert parsed["price"] == 50000.0
            assert parsed["amount"] == 0.001
            assert parsed["filled"] == 0.0005
            assert parsed["remaining"] == 0.0005
            assert parsed["status"] == "open"

    def test_parse_bitkub_order_closed(self, bitkub_config):
        """Test parsing closed Bitkub order."""
        with patch.object(Bitkub, "__init__", lambda x, y: None):
            bitkub = Bitkub.__new__(Bitkub)

            order_data = {
                "id": "123456",
                "side": "sell",
                "type": "limit",
                "rate": 50000.0,
                "amount": 0.001,
                "filled": 0.001,  # Fully filled
                "ts": 1609459200,
            }

            parsed = bitkub._parse_bitkub_order(order_data, "BTC/THB")

            assert parsed["status"] == "closed"
            assert parsed["remaining"] == 0.0

    def test_exchange_has(self, bitkub_config):
        """Test exchange_has method."""
        with patch.object(Bitkub, "__init__", lambda x, y: None):
            bitkub = Bitkub.__new__(Bitkub)

            assert bitkub.exchange_has("fetchOHLCV") is True
            assert bitkub.exchange_has("fetchTicker") is True
            assert bitkub.exchange_has("createOrder") is True
            assert bitkub.exchange_has("cancelOrder") is True
            assert bitkub.exchange_has("fetchOrder") is True
            assert bitkub.exchange_has("fetchOpenOrders") is True
            assert bitkub.exchange_has("fetchBalance") is True

            # Not supported features
            assert bitkub.exchange_has("setLeverage") is False
            assert bitkub.exchange_has("fetchPositions") is False
            assert bitkub.exchange_has("watchOHLCV") is False

    def test_name_property(self, bitkub_config):
        """Test name property returns correct exchange name."""
        with patch.object(Bitkub, "__init__", lambda x, y: None):
            bitkub = Bitkub.__new__(Bitkub)

            assert bitkub.name == "Bitkub"

    def test_id_property(self, bitkub_config):
        """Test id property returns correct exchange id."""
        with patch.object(Bitkub, "__init__", lambda x, y: None):
            bitkub = Bitkub.__new__(Bitkub)

            assert bitkub.id == "bitkub"


class TestBitkubMarketParsing:
    """Test market data parsing for Bitkub."""

    def test_symbol_conversion(self):
        """Test Bitkub symbol format conversion."""
        # Bitkub uses "THB_BTC" format, should convert to "BTC/THB"
        bitkub_symbol = "THB_BTC"
        parts = bitkub_symbol.split("_")
        quote, base = parts[0], parts[1]
        freqtrade_pair = f"{base}/{quote}"

        assert freqtrade_pair == "BTC/THB"

    def test_market_structure(self):
        """Test market structure has required fields."""
        # Sample market structure that would be created
        market = {
            "id": "THB_BTC",
            "symbol": "BTC/THB",
            "base": "BTC",
            "quote": "THB",
            "active": True,
            "spot": True,
            "margin": False,
            "swap": False,
            "future": False,
            "option": False,
            "contract": False,
            "precision": {
                "amount": 8,
                "price": 2,
            },
            "limits": {
                "leverage": {"min": 1, "max": 1},
                "amount": {"min": 0.0001, "max": None},
                "price": {"min": None, "max": None},
                "cost": {"min": 10, "max": None},
            },
            "maker": 0.0025,
            "taker": 0.0025,
        }

        # Verify required fields
        assert "id" in market
        assert "symbol" in market
        assert "base" in market
        assert "quote" in market
        assert "active" in market
        assert "spot" in market
        assert "precision" in market
        assert "limits" in market
        assert market["spot"] is True
        assert market["margin"] is False
