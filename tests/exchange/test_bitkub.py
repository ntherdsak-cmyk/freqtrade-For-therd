"""Tests for Bitkub exchange adapter."""

from unittest.mock import MagicMock, patch

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.bitkub import Bitkub, _BitkubMockCcxt


class TestBitkub:
    """Tests for Bitkub class."""

    def test_bitkub_requires_dry_run(self, default_conf, mocker):
        """Test that Bitkub raises error when dry_run is False."""
        default_conf["dry_run"] = False
        default_conf["exchange"]["name"] = "bitkub"

        with pytest.raises(OperationalException, match="dry-run mode"):
            Bitkub(config=default_conf)

    def test_bitkub_init_with_dry_run(self, default_conf, mocker):
        """Test that Bitkub initializes correctly in dry_run mode."""
        default_conf["dry_run"] = True
        default_conf["exchange"]["name"] = "bitkub"

        # Mock the bitkub-python Client class
        mock_client_instance = MagicMock()
        mock_client_instance.fetch_symbols.return_value = {
            "result": [
                {"symbol": "THB_BTC"},
                {"symbol": "THB_ETH"},
            ]
        }

        mock_client_class = MagicMock(return_value=mock_client_instance)

        with patch.dict("sys.modules", {"bitkub": MagicMock(Client=mock_client_class)}):
            exchange = Bitkub(config=default_conf, validate=False)

            assert exchange.name == "Bitkub"
            assert exchange.id == "bitkub"
            assert exchange._bitkub_client is not None

    def test_bitkub_init_without_bitkub_library(self, default_conf, mocker, caplog):
        """Test that Bitkub handles missing bitkub-python library gracefully."""
        default_conf["dry_run"] = True
        default_conf["exchange"]["name"] = "bitkub"

        # Simulate bitkub module not being available
        with patch.dict("sys.modules", {"bitkub": None}):
            exchange = Bitkub(config=default_conf, validate=False)

            assert exchange.name == "Bitkub"
            # Should still work even without the bitkub library


class TestBitkubMockCcxt:
    """Tests for _BitkubMockCcxt class."""

    def test_mock_ccxt_init(self):
        """Test _BitkubMockCcxt initialization."""
        mock_ccxt = _BitkubMockCcxt()

        assert mock_ccxt.name == "Bitkub"
        assert mock_ccxt.id == "bitkub"
        assert "1h" in mock_ccxt.timeframes
        assert mock_ccxt.has["fetchTicker"] is True
        assert mock_ccxt.has["fetchOHLCV"] is True

    def test_mock_ccxt_with_client(self):
        """Test _BitkubMockCcxt with bitkub client."""
        mock_client = MagicMock()
        mock_client.fetch_symbols.return_value = {
            "result": [
                {"symbol": "THB_BTC"},
                {"symbol": "THB_ETH"},
            ]
        }

        mock_ccxt = _BitkubMockCcxt(mock_client)

        assert "BTC/THB" in mock_ccxt.markets
        assert "ETH/THB" in mock_ccxt.markets
        assert mock_ccxt.markets["BTC/THB"]["base"] == "BTC"
        assert mock_ccxt.markets["BTC/THB"]["quote"] == "THB"

    def test_mock_ccxt_default_markets(self):
        """Test _BitkubMockCcxt default markets when API fails."""
        mock_client = MagicMock()
        mock_client.fetch_symbols.side_effect = Exception("API Error")

        mock_ccxt = _BitkubMockCcxt(mock_client)

        # Should fall back to default markets
        assert "BTC/THB" in mock_ccxt.markets

    def test_mock_ccxt_fetch_ticker(self):
        """Test _BitkubMockCcxt fetch_ticker method."""
        mock_client = MagicMock()
        mock_client.fetch_symbols.return_value = {"result": []}
        mock_client.fetch_tickers.return_value = {
            "THB_BTC": {
                "high24hr": 1500000,
                "low24hr": 1400000,
                "highestBid": 1450000,
                "lowestAsk": 1455000,
                "last": 1452000,
                "baseVolume": 100,
                "quoteVolume": 145200000,
            }
        }

        mock_ccxt = _BitkubMockCcxt(mock_client)
        ticker = mock_ccxt.fetch_ticker("BTC/THB")

        assert ticker["symbol"] == "BTC/THB"
        assert ticker["high"] == 1500000
        assert ticker["low"] == 1400000
        assert ticker["bid"] == 1450000
        assert ticker["ask"] == 1455000
        assert ticker["last"] == 1452000

    def test_mock_ccxt_fetch_ticker_fallback(self):
        """Test _BitkubMockCcxt fetch_ticker fallback when API fails."""
        mock_ccxt = _BitkubMockCcxt()
        ticker = mock_ccxt.fetch_ticker("BTC/THB")

        assert ticker["symbol"] == "BTC/THB"
        assert "timestamp" in ticker

    def test_mock_ccxt_fetch_ohlcv(self):
        """Test _BitkubMockCcxt fetch_ohlcv method."""
        mock_client = MagicMock()
        mock_client.fetch_symbols.return_value = {"result": []}
        mock_client.fetch_trading_view_history.return_value = {
            "t": [1700000000, 1700003600, 1700007200],
            "o": [1450000, 1455000, 1452000],
            "h": [1460000, 1458000, 1457000],
            "l": [1445000, 1450000, 1448000],
            "c": [1455000, 1452000, 1454000],
            "v": [100.5, 150.2, 80.3],
        }

        mock_ccxt = _BitkubMockCcxt(mock_client)
        ohlcv = mock_ccxt.fetch_ohlcv("BTC/THB", "1h")

        assert len(ohlcv) == 3
        assert ohlcv[0][0] == 1700000000 * 1000  # timestamp in ms
        assert ohlcv[0][1] == 1450000  # open
        assert ohlcv[0][2] == 1460000  # high
        assert ohlcv[0][3] == 1445000  # low
        assert ohlcv[0][4] == 1455000  # close
        assert ohlcv[0][5] == 100.5  # volume

    def test_mock_ccxt_fetch_ohlcv_empty(self):
        """Test _BitkubMockCcxt fetch_ohlcv returns empty list when API unavailable."""
        mock_ccxt = _BitkubMockCcxt()
        ohlcv = mock_ccxt.fetch_ohlcv("BTC/THB", "1h")

        assert ohlcv == []

    def test_mock_ccxt_fetch_tickers(self):
        """Test _BitkubMockCcxt fetch_tickers method."""
        mock_ccxt = _BitkubMockCcxt()
        mock_ccxt.markets = {"BTC/THB": {}, "ETH/THB": {}}

        tickers = mock_ccxt.fetch_tickers(["BTC/THB", "ETH/THB"])

        assert "BTC/THB" in tickers
        assert "ETH/THB" in tickers

    def test_mock_ccxt_fetch_order_book(self):
        """Test _BitkubMockCcxt fetch_order_book method."""
        mock_client = MagicMock()
        mock_client.fetch_symbols.return_value = {"result": []}
        mock_client.fetch_bids.return_value = {
            "result": [
                {"price": 1450000, "amount": 0.5},
                {"price": 1449000, "amount": 1.0},
            ]
        }
        mock_client.fetch_asks.return_value = {
            "result": [
                {"price": 1455000, "amount": 0.3},
                {"price": 1456000, "amount": 0.7},
            ]
        }

        mock_ccxt = _BitkubMockCcxt(mock_client)
        order_book = mock_ccxt.fetch_order_book("BTC/THB")

        assert order_book["symbol"] == "BTC/THB"
        assert len(order_book["bids"]) == 2
        assert len(order_book["asks"]) == 2
        assert order_book["bids"][0] == [1450000, 0.5]

    def test_mock_ccxt_fetch_balance(self):
        """Test _BitkubMockCcxt fetch_balance method."""
        mock_ccxt = _BitkubMockCcxt()
        balance = mock_ccxt.fetch_balance()

        assert "info" in balance
        assert "free" in balance
        assert "used" in balance
        assert "total" in balance

    def test_mock_ccxt_load_markets(self):
        """Test _BitkubMockCcxt load_markets method."""
        mock_client = MagicMock()
        mock_client.fetch_symbols.return_value = {
            "result": [
                {"symbol": "THB_BTC"},
            ]
        }

        mock_ccxt = _BitkubMockCcxt(mock_client)

        # Clear markets and reload
        mock_ccxt.markets = {}
        markets = mock_ccxt.load_markets(reload=True)

        assert "BTC/THB" in markets

    def test_mock_ccxt_fetch_l2_order_book(self):
        """Test _BitkubMockCcxt fetch_l2_order_book is alias for fetch_order_book."""
        mock_ccxt = _BitkubMockCcxt()
        order_book = mock_ccxt.fetch_l2_order_book("BTC/THB")

        assert "bids" in order_book
        assert "asks" in order_book
