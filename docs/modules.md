# Module Overview

This document outlines the main modules of the Hyper-Trading Automation project and their primary inputs and outputs.

## hypertrader.bot
Runs one iteration of the trading pipeline. Accepts command line arguments or values from `config.yaml` and writes a JSON signal containing action, volume, and risk parameters.

## hypertrader.data
Utilities for retrieving market and macroeconomic data. Functions return pandas objects ready for feature engineering.

Includes on-chain helpers such as Ethereum gas fee retrieval.

## hypertrader.strategies
Rule-based and machine-learning strategies that transform features into trading signals, now supporting on-chain activity and order book imbalance filters.

## hypertrader.utils
Helper routines for indicators, risk management, configuration loading, logging, and retry logic. These functions are used across the project to maintain consistency.
