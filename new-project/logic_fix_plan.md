
# Bitcoin Simulation Logic Fix Plan

## Identified Logic Issues

Based on my analysis of `leverage_sim.py`, here are the critical logic problems that need fixing:

### 1. **Incorrect Interest Calculation**
**Current Problem**: 
```python
interest = loan * state["cycle"] * 0  # placeholder (simple loop-year calc)
payoff = loan + loan * params["loan_rate"]  # 1-yr APR lump
```
- Interest is always calculated as zero
- Payoff assumes exactly 1 year regardless of actual cycle time

**Impact**: Massively underestimates borrowing costs, making strategy appear more profitable than reality

### 2. **Unrealistic Price Movement Model**
**Current Problem**:
- Every cycle assumes exactly +$30K price increase
- No consideration of time required for such moves
- No volatility or realistic market dynamics

**Impact**: Overly optimistic returns that don't reflect Bitcoin's actual price behavior

### 3. **Margin Call Timing Logic Error**
**Current Problem**:
- Assumes margin call happens immediately at cycle start
- Uses fitted 99th percentile drop without considering probability
- No realistic recovery time modeling

**Impact**: May underestimate actual margin call risk and timing

### 4. **Reserve Management Inconsistency**
**Current Problem**:
```python
state["collat_btc"] = params["start_collateral_btc"]  # reset reserve
```
- Reserve resets to initial amount regardless of accumulated BTC
- Doesn't scale reserve with growing portfolio

**Impact**: Fails to properly size risk management as portfolio grows

### 5. **Drawdown Model Overfitting Risk**
**Current Problem**:
- Complex curve fitting with limited validation
- May not generalize to future market conditions
- No confidence intervals or uncertainty modeling

**Impact**: False precision in risk estimates

## Fix Implementation Plan

### Phase 1: Fix Interest Calculation Logic

#### Step 1.1: Add Realistic Time Modeling
```python
# Replace fixed $30K jumps with realistic time-based price modeling
def calculate_cycle_time(start_price, target_price, historical_data):
    """Calculate realistic time for price appreciation based on historical patterns"""
    # Analyze historical time for similar price moves
    # Return expected days + volatility
```

#### Step 1.2: Implement Proper Interest Accrual
```python
def calculate_interest(principal, rate, days):
    """Calculate compound interest for actual holding period"""
    if days <= 0:
        return 0
    daily_rate = rate / 365
    return principal * ((1 + daily_rate) ** days - 1)
```

### Phase 2: Implement Realistic Price Movement

#### Step 2.1: Historical Price Movement Analysis
```python
def analyze_price_movements(price_series, jump_amount):
    """Analyze how long Bitcoin historically takes for specific price moves"""
    # Find all instances where price increased by jump_amount
    # Calculate distribution of time periods
    # Return probability-weighted time estimates
```

#### Step 2.2: Monte Carlo Price Simulation
```python
def simulate_price_path(start_price, target_price, volatility):
    """Generate realistic price path using historical volatility"""
    # Use geometric Brownian motion or similar
    # Include realistic drawdowns during appreciation
    # Return full price path, not just endpoints
```

### Phase 3: Fix Margin Call Logic

#### Step 3.1: Dynamic Margin Call Modeling
```python
def simulate_margin_calls(price_path, entry_price, loan_amount, collateral_btc):
    """Simulate when margin calls actually occur during price path"""
    margin_calls = []
    for day, price in enumerate(price_path):
        ltv = loan_amount / (collateral_btc * price)
        if ltv >= 0.90:
            margin_calls.append((day, price, ltv))
    return margin_calls
```

#### Step 3.2: Probabilistic Drawdown Application
```python
def apply_realistic_drawdowns(price_path, drawdown_model):
    """Apply drawdowns probabilistically, not deterministically"""
    # Don't assume worst-case drawdown happens every cycle
    # Use Monte Carlo to sample from drawdown distribution
    # Consider correlation with market conditions
```

### Phase 4: Fix Reserve Management

#### Step 4.1: Proportional Reserve Scaling
```python
def calculate_optimal_reserve(total_btc_holdings, current_price, target_ltv=0.50):
    """Scale reserve size with portfolio growth"""
    # Reserve should grow with portfolio
    # Consider optimal capital allocation
    # Balance safety vs. opportunity cost
```

#### Step 4.2: Dynamic Risk Management
```python
def adjust_position_sizing(reserve_btc, market_volatility, historical_drawdowns):
    """Adjust loan sizing based on current market conditions"""
    # Scale loan size with volatility regime
    # Consider recent drawdown history
    # Implement position sizing that adapts to risk
```

### Phase 5: Improve Drawdown Model Robustness

#### Step 5.1: Model Validation and Uncertainty
```python
def validate_drawdown_model(model, historical_data, test_ratio=0.3):
    """Cross-validate drawdown model with out-of-sample testing"""
    # Split data into train/test
    # Calculate prediction intervals
    # Test model stability across different time periods
```

#### Step 5.2: Ensemble Modeling
```python
def create_drawdown_ensemble(price_data):
    """Use multiple models to estimate drawdown risk"""
    # Power law model (current)
    # Empirical percentiles
    # Volatility-based estimates
    # Return confidence intervals, not point estimates
```

### Phase 6: Add Realistic Market Constraints

#### Step 6.1: Transaction Costs and Slippage
```python
def apply_transaction_costs(btc_amount, price, transaction_type='buy'):
    """Apply realistic trading costs"""
    # Exchange fees (0.1-0.5%)
    # Bid-ask spread impact
    # Slippage for large orders
    # Time delays for execution
```

#### Step 6.2: Liquidity and Market Impact
```python
def check_market_liquidity(btc_amount, current_price):
    """Ensure trades are realistic given market depth"""
    # Consider exchange order books
    # Factor in market impact for large trades
    # Add execution time delays
```

## Implementation Priority

### Critical Fixes (Must Fix):
1. **Interest calculation** - Currently zero, completely invalidates results
2. **Realistic time modeling** - $30K jumps don't happen overnight
3. **Proper margin call timing** - Current logic is too pessimistic/unrealistic

### Important Fixes (Should Fix):
4. **Reserve scaling** - Needed for accurate long-term projections
5. **Drawdown model validation** - Reduce overfitting risk

### Enhancement Fixes (Nice to Have):
6. **Transaction costs** - More realistic P&L
7. **Monte Carlo scenarios** - Better risk assessment

## Testing Strategy

### Step 1: Unit Tests for Each Fix
```python
def test_interest_calculation():
    # Test various loan amounts, rates, time periods
    # Verify compound interest formula correctness

def test_price_movement_realism():
    # Verify generated price paths match historical statistics
    # Check that time estimates are reasonable

def test_margin_call_logic():
    # Verify margin calls trigger at correct LTV levels
    # Test edge cases and boundary conditions
```

### Step 2: Integration Testing
```python
def test_full_simulation_scenarios():
    # Run simulation with different market conditions
    # Compare results to historical back-tests
    # Verify results make intuitive sense
```

### Step 3: Sensitivity Analysis
```python
def test_parameter_sensitivity():
    # Vary key assumptions (interest rates, drawdown model, etc.)
    # Measure impact on final results
    # Identify which assumptions matter most
```

## Expected Impact of Fixes

### Before Fixes:
- Unrealistically optimistic returns
- Zero interest costs
- Deterministic worst-case scenarios
- Fixed reserve ratios

### After Fixes:
- Realistic borrowing costs included
- Probabilistic risk modeling
- Time-aware price movements
- Adaptive risk management
- Validated model assumptions

## Success Metrics

✅ **Interest costs realistic**: Borrowing costs match market rates
✅ **Time modeling accurate**: Cycle durations match historical patterns  
✅ **Margin calls probabilistic**: Not every cycle hits worst-case
✅ **Reserve scaling logical**: Risk management grows with portfolio
✅ **Model validation passes**: Out-of-sample testing shows stability
✅ **Results make intuitive sense**: Strategy performance is believable

This plan addresses the core logic flaws that make the current simulation unrealistically optimistic and provides a roadmap to create a more accurate, robust lending strategy analysis.
