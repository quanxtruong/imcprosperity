from typing import Dict, List, Any
import statistics
import json
import math
import numpy as np
import pandas as pd
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 31,
        "reversion_beta": -0.2,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "window_size": 500,
        "position_limit": 50,
    }
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50, 
            Product.KELP: 50,
            Product.SQUID_INK: 50
        }

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def KELP_fair_value(self, order_depth: OrderDepth, trader_data) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if trader_data.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = trader_data["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            
            # Step 3: Store midprice history
            trader_data.setdefault("KELP_midprice_history", [])
            trader_data["KELP_midprice_history"].append(mmmid_price)
            if len(trader_data["KELP_midprice_history"]) > 10:
                trader_data["KELP_midprice_history"].pop(0)

            # Step 4: Calculate volatility
            vol = self.ewma_volatility(trader_data["KELP_midprice_history"], span=5)
            vol_threshold = 0.003

            # Step 5: Adjust beta dynamically
            base_beta = self.params[Product.KELP]["reversion_beta"]
            if vol > vol_threshold:
                adjusted_beta = base_beta * 1.75
            else:
                adjusted_beta = base_beta * 0.75

            # Optionally store for tracking
            trader_data["KELP_last_beta"] = adjusted_beta

            # Step 6: Predict fair value
            if trader_data.get("KELP_last_price") is not None:
                last_price = trader_data["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * adjusted_beta
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            trader_data["KELP_last_price"] = mmmid_price
            return fair
        return None

    def ewma_volatility(self, prices, span=10):
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return pd.Series(returns).ewm(span=span).std().iloc[-1]

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def trade_squid_ink(self, state: TradingState, trader_data: Dict) -> List[Order]:
        product = Product.SQUID_INK
        orders = []
        
        if product not in state.order_depths:
            return orders
            
        # Initialize price history for this product if needed
        if "price_history" not in trader_data:
            trader_data["price_history"] = {}
            
        if product not in trader_data["price_history"]:
            trader_data["price_history"][product] = []
            
        order_depth = state.order_depths[product]
        
        # Skip if market is one-sided
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        # Get market data
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_volume = order_depth.buy_orders[best_bid]
        best_ask_volume = order_depth.sell_orders[best_ask]
        mid_price = (best_bid + best_ask) / 2
        
        # Get current position
        current_position = state.position.get(product, 0)
        position_limit = self.LIMIT[product]
        
        # Calculate available room to trade
        room_to_buy = position_limit - current_position
        room_to_sell = position_limit + current_position
        
        # Update price history
        trader_data["price_history"][product].append(mid_price)
        
        # Set window size for SQUID_INK
        window_size = self.params[Product.SQUID_INK]["window_size"]
        
        # Trim price history to window size
        if len(trader_data["price_history"][product]) > window_size:
            trader_data["price_history"][product] = trader_data["price_history"][product][-window_size:]
            
        # SQUID_INK strategy - mean reversion with volatility awareness
        if len(trader_data["price_history"][product]) >= 20:
            moving_avg = statistics.mean(trader_data["price_history"][product])
            std_dev = statistics.stdev(trader_data["price_history"][product]) if len(trader_data["price_history"][product]) > 1 else 0
            
            # Skip if no meaningful statistics
            if std_dev == 0:
                return orders
            
            # Analyze recent volatility (last 20 vs previous 20)
            if len(trader_data["price_history"][product]) >= 40:
                recent_prices = trader_data["price_history"][product][-20:]
                previous_prices = trader_data["price_history"][product][-40:-20]
                recent_std_dev = statistics.stdev(recent_prices)
                previous_std_dev = statistics.stdev(previous_prices)
                
                # If recent volatility is increasing rapidly, be more cautious
                volatility_change = recent_std_dev / previous_std_dev if previous_std_dev > 0 else 1.0
                
                # Adjust std_multiplier based on volatility trend
                std_multiplier = 1.5
                if volatility_change > 1.5:
                    # Volatility increasing - widen bands
                    std_multiplier = 1.75
                elif volatility_change < 0.7:
                    # Volatility decreasing - narrow bands
                    std_multiplier = 1.25
            else:
                std_multiplier = 1.5
            
            # Calculate order book imbalance
            total_bid_volume = sum(-qty for qty in order_depth.buy_orders.values())
            total_ask_volume = sum(qty for qty in order_depth.sell_orders.values())
            
            book_imbalance = 0
            if total_bid_volume + total_ask_volume > 0:
                book_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # Adjust thresholds based on order book imbalance
            imbalance_adjustment = 0.2 * std_dev * book_imbalance
            
            upper_threshold = moving_avg + (std_multiplier * std_dev) + imbalance_adjustment
            lower_threshold = moving_avg - (std_multiplier * std_dev) + imbalance_adjustment
            
            logger.print(f"SQUID_INK: Price={mid_price:.2f}, MA={moving_avg:.2f}, StdDev={std_dev:.2f}, Imbalance={book_imbalance:.2f}, StdMult={std_multiplier:.2f}")
            
            # Trading logic with aggressive mean reversion
            if mid_price > upper_threshold:
                # Calculate deviation in std dev units
                deviation = (mid_price - moving_avg) / std_dev
                
                # More aggressive for larger deviations, but cap at 1.0
                position_scalar = min(1.0, 0.4 + deviation * 0.25)
                
                # Consider current position in decision - reduce size when we already have a large position
                position_factor = max(0.2, 1.0 - (current_position / position_limit) * 0.7)
                
                sell_quantity = min(best_bid_volume, math.ceil(room_to_sell * position_scalar * position_factor))
                
                if sell_quantity > 0:
                    logger.print(f"SQUID_INK: SELL {sell_quantity}x {best_bid} (high price, deviation: {deviation:.2f})")
                    orders.append(Order(product, best_bid, -sell_quantity))
            
            elif mid_price < lower_threshold:
                # Calculate deviation in std dev units
                deviation = (moving_avg - mid_price) / std_dev
                
                # More aggressive for larger deviations, but cap at 1.0
                position_scalar = min(1.0, 0.4 + deviation * 0.25)
                
                # Consider current position in decision - reduce size when we already have a large position
                position_factor = max(0.2, 1.0 + (current_position / position_limit) * 0.7)
                
                buy_quantity = min(-best_ask_volume, math.ceil(room_to_buy * position_scalar * position_factor))
                
                if buy_quantity > 0:
                    logger.print(f"SQUID_INK: BUY {buy_quantity}x {best_ask} (low price, deviation: {deviation:.2f})")
                    orders.append(Order(product, best_ask, buy_quantity))
        
        return orders

    def run(self, state: TradingState):
        # Initialize trader data
        trader_data = {}
        if state.traderData:
            try:
                if state.traderData.startswith("{"):  # Check if it's JSON
                    trader_data = json.loads(state.traderData)
                else:
                    trader_data = jsonpickle.decode(state.traderData)
            except:
                # If there's an error, start with a fresh trader_data
                trader_data = {}
        
        result = {}
        conversions = 0

        # Handle SQUID_INK trading
        if Product.SQUID_INK in state.order_depths:
            squid_ink_orders = self.trade_squid_ink(state, trader_data)
            if squid_ink_orders:
                result[Product.SQUID_INK] = squid_ink_orders

        # Handle RAINFOREST_RESIN trading
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            rainforest_resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    rainforest_resin_position,
                )
            )
            rainforest_resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    rainforest_resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            rainforest_resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                rainforest_resin_take_orders + rainforest_resin_clear_orders + rainforest_resin_make_orders
            )

        # Handle KELP trading
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], trader_data
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )

        # Save trader data - decide whether to use json or jsonpickle based on input format
        if state.traderData and state.traderData.startswith("{"):
            traderData = json.dumps(trader_data)
        else:
            traderData = jsonpickle.encode(trader_data)
            
        # Default conversions to 1 as in the original KELP/RAINFOREST_RESIN strategy
        conversions = 1
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData