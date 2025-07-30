
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

**Real Contract Analysis (Figure Lending $30K @ 12.615% APR)**:
- **Daily Interest Accrual**: Interest accrues daily at APR/365 (or 366 in leap years)
- **Interest-Only Monthly Payments**: $295.95/month for 11 months
- **Balloon Payment**: Full $30,300 principal + accrued interest at month 12
- **No Prepayment Penalty**: Can pay off early without fees
- **Loan-to-Value (LTV) Management**:
  - Maximum LTV: 75% (loan baseline)
  - Margin call trigger: 85% LTV (48 hours to cure)
  - Force liquidation: 90% LTV (immediate)
  - Collateral release possible: <35% LTV for 7+ days

**Key Insight**: The simulation needs to account for monthly interest payments PLUS the risk of forced liquidation during market downturns, not just end-of-cycle interest.

**Impact**: Current model massively underestimates both borrowing costs AND liquidation risk

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

#### Step 1.2: Implement Realistic Interest Structure (Based on Figure Lending Contract)
```python
def calculate_monthly_payment(principal, apr):
    """Calculate monthly interest-only payment"""
    # For $10K loan at 11.5% APR (minimum available rate)
    monthly_rate = apr / 12
    return principal * monthly_rate

def calculate_daily_interest_accrual(principal, apr, days):
    """Calculate daily compound interest as per contract terms"""
    daily_rate = apr / 365  # Contract specifies 365-day year
    return principal * ((1 + daily_rate) ** days - 1)

def check_ltv_triggers(loan_balance, collateral_value):
    """Monitor LTV levels for margin calls and liquidations"""
    ltv = loan_balance / collateral_value
    
    if ltv >= 0.90:
        return "FORCE_LIQUIDATION"  # Immediate sale
    elif ltv >= 0.85:
        return "MARGIN_CALL"  # 48 hours to add collateral
    elif ltv <= 0.35:
        return "COLLATERAL_RELEASE_ELIGIBLE"  # Can withdraw excess
    else:
        return "NORMAL"

def simulate_monthly_interest_burden(principal, apr, btc_price_path):
    """Simulate the cash flow impact of monthly interest payments"""
    monthly_payment = calculate_monthly_payment(principal, apr)
    # This requires selling BTC each month to make payments
    # Reduces available collateral and increases liquidation risk
    return monthly_payment
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

### Phase 3: Implement Realistic LTV Management (Critical New Finding)

#### Step 3.0: Model Monthly Interest Payment Impact
```python
def simulate_interest_payment_impact(btc_holdings, monthly_payment, btc_price):
    """Model how monthly interest payments reduce collateral"""
    # Each month, must sell BTC to make interest payment
    btc_sold_for_interest = monthly_payment / btc_price
    remaining_btc = btc_holdings - btc_sold_for_interest
    
    # This reduces collateral, increasing LTV ratio over time
    # Creates compounding liquidation risk
    return remaining_btc, btc_sold_for_interest

def model_ltv_drift(initial_ltv, monthly_payment, btc_volatility, time_months):
    """Model how LTV drifts due to interest payments + volatility"""
    # LTV increases from interest payments (selling collateral)
    # LTV fluctuates with BTC price volatility
    # Need to model probability of hitting 85%/90% triggers
```

#### Step 3.1: Daily LTV Monitoring System
```python
def daily_ltv_check(loan_balance, collateral_btc, btc_price, day):
    """Check LTV status every day of simulation"""
    collateral_value = collateral_btc * btc_price
    current_ltv = loan_balance / collateral_value
    
    # Log all margin call events, not just end-of-cycle
    if current_ltv >= 0.90:
        return trigger_force_liquidation(collateral_btc, loan_balance)
    elif current_ltv >= 0.85:
        return trigger_margin_call(day)
    
    return current_ltv

def trigger_margin_call(day):
    """Model 48-hour cure period for margin calls"""
    # User has 48 hours to:
    # 1. Add more BTC collateral, OR
    # 2. Partially pay down loan
    # If neither happens -> force liquidation
    cure_deadline = day + 2
    return {"type": "margin_call", "cure_deadline": cure_deadline}
```

### Phase 4: Fix Margin Call Logic

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

## Key Contract Terms That Must Be Modeled

Based on the Figure Lending contract analysis:

### Critical Financial Terms:
- **Principal**: $10,000 (minimum loan amount available)
- **APR**: 11.5% (minimum rate for $10K loan, not 12.615% from $30K contract)
- **Term**: 12 months maximum
- **Monthly Payment**: ~$95.83 (interest-only: $10,000 × 11.5% ÷ 12)
- **Origination Fee**: ~$333 (estimated based on contract scaling, added to principal)

**Note**: The $30K contract at 12.615% APR provides the operational framework (LTV ratios, margin call procedures, etc.), but the financial terms scale down for the minimum $10K loan at 11.5% APR.

### LTV Management Rules:
- **Baseline LTV**: 75% maximum
- **Margin Call**: 85% LTV (48-hour cure period)
- **Force Liquidation**: 90% LTV (immediate, 2% processing fee)
- **Collateral Release**: Available when <35% LTV for 7+ consecutive days

### Daily Operational Model Required:
1. **Daily interest accrual** on outstanding balance
2. **Monthly interest payments** (requires selling BTC)
3. **Daily LTV monitoring** for margin calls
4. **Cure period management** (48 hours to respond to margin calls)
5. **Liquidation modeling** (immediate at 90% LTV)

## Expected Impact of Fixes

### Before Fixes:
- Unrealistically optimistic returns
- Zero interest costs
- Deterministic worst-case scenarios
- Fixed reserve ratios
- No LTV management
- No monthly cash flow requirements

### After Fixes:
- **Realistic borrowing costs**: 11.5% APR with monthly payments ($95.83/month)
- **LTV risk management**: Daily monitoring with margin calls
- **Cash flow modeling**: Monthly BTC sales for interest payments
- **Liquidation risk**: Proper 85%/90% LTV trigger modeling
- **Time-aware price movements**: Accounts for actual market dynamics
- **Validated model assumptions**: Based on real contract terms

### Critical Realization:
The monthly interest payments create a **compounding liquidation risk** because:
1. Each month requires selling BTC to make interest payment
2. This reduces collateral, increasing LTV ratio
3. During bear markets, this accelerates toward liquidation
4. Strategy may be fundamentally unprofitable due to this dynamic

## Success Metrics

✅ **Interest costs realistic**: Borrowing costs match market rates
✅ **Time modeling accurate**: Cycle durations match historical patterns  
✅ **Margin calls probabilistic**: Not every cycle hits worst-case
✅ **Reserve scaling logical**: Risk management grows with portfolio
✅ **Model validation passes**: Out-of-sample testing shows stability
✅ **Results make intuitive sense**: Strategy performance is believable

This plan addresses the core logic flaws that make the current simulation unrealistically optimistic and provides a roadmap to create a more accurate, robust lending strategy analysis.
