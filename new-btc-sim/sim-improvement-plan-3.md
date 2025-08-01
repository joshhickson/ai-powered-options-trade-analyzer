
# Pure Forward Bitcoin Simulation - Technical Implementation Plan

## 🎯 **OBJECTIVE: Pure Forward-Looking Simulation**

**Problem Statement**: Current simulation in `new-btc-sim/leverage_war_days.py` has fundamental time direction issues that create impossible market scenarios. We need a pure forward-looking simulation that starts from current market conditions and projects realistic future scenarios.

**Solution**: Implement a Monte Carlo-based forward simulation using current market price as the starting point, with realistic scenario modeling based on historical volatility patterns.

## 📁 **KEY FILES IN SCOPE**
- **Primary Implementation**: `new-btc-sim/leverage_war_days.py` (current broken simulation)
- **Contract Reference**: `new-btc-sim/Loan contract ocr.md` (Figure Lending LLC terms)
- **Strategy Logic**: `new-btc-sim/Loan Mermaids.md` (mermaid flowcharts)
- **Legacy Examples**: `old-projects/leverage_sim.py`, `old-projects/leverage_sim2.py`, `old-projects/leverage_sim3.py`
- **Export Directory**: `new-btc-sim/exports/` (simulation results)

---

## 🔍 **ROOT CAUSE ANALYSIS**

### **Current Issues Identified in `new-btc-sim/leverage_war_days.py`**
1. **Time Direction Confusion**: Lines 172-176 use current prices to buy initial BTC, then line 179 immediately processes 2-year-old historical prices
2. **Impossible Market Scenarios**: Starting at $118K then immediately facing $29K prices creates unrealistic liquidation scenarios  
3. **Single Cycle Execution**: Only runs one loan cycle due to broken exit condition logic in `run()` method (lines 140-190)
4. **Missing Profit-Taking**: No cycle completion despite price movements that should trigger exits in `start_new_cycle()` method

### **Why Current Approach Fails in `leverage_war_days.py`**
```python
# BROKEN: Current hybrid approach in leverage_war_days.py lines 172-179
start_price = price_series.iloc[-1]  # Uses current price ($118K) - line 172
initial_btc = INITIAL_USD_CAPITAL / start_price  # Buys 0.25 BTC - line 175
forward_series = price_series.iloc[::-1]  # Processes old prices ($29K) - line 179
# Result: Immediate impossible price crash
```

---

## 🚀 **PURE FORWARD SIMULATION ARCHITECTURE**

### **Core Design Principles**
1. **Current Market Reality**: Start with actual current BTC price and holdings
2. **Forward Time Flow**: Generate realistic future price paths, never look backward
3. **Scenario-Based Modeling**: Use Monte Carlo with historically-informed scenarios
4. **Contract Compliance**: All loan parameters must match Figure Lending LLC terms exactly

### **Implementation Strategy**

#### **Phase 1: Current Market Initialization**
**Target File**: `new-btc-sim/leverage_war_days.py` - Replace `get_historical_data()` function (lines 42-66)
```python
def initialize_current_market_state():
    """Initialize simulation with real current market conditions"""
    current_price = fetch_live_btc_price()  # Real current market price
    initial_usd = 30000.0
    initial_btc = initial_usd / current_price * (1 - TRADING_FEE_PERCENT)
    
    return {
        'current_price': current_price,
        'total_btc': initial_btc,
        'simulation_start_date': datetime.now(),
        'market_regime': classify_current_market_regime(current_price)
    }
```

#### **Phase 2: Forward Price Path Generation**
```python
def generate_forward_price_scenarios(start_price, simulation_days=730):
    """Generate realistic forward-looking price scenarios"""
    scenarios = {
        'bull_market': {
            'probability': 0.20,
            'annual_return': 1.5,  # 150% annual return
            'volatility': 0.8
        },
        'bear_market': {
            'probability': 0.25,
            'annual_return': -0.6,  # -60% annual return
            'volatility': 1.2
        },
        'sideways_market': {
            'probability': 0.35,
            'annual_return': 0.1,  # 10% annual return
            'volatility': 0.6
        },
        'crash_recovery': {
            'probability': 0.20,
            'annual_return': 0.3,  # 30% annual return
            'volatility': 1.5
        }
    }
    
    # Generate weighted scenario-based price paths
    return create_monte_carlo_price_paths(start_price, scenarios, simulation_days)
```

#### **Phase 3: Loan Cycle Logic Redesign**
```python
class ForwardLoanSimulator:
    def __init__(self):
        # CONTRACT COMPLIANCE (Figure Lending LLC terms)
        self.LOAN_APR = 0.12615
        self.LTV_BASELINE = 0.75
        self.MARGIN_CALL_LTV = 0.85
        self.LIQUIDATION_LTV = 0.90
        self.PROFIT_TAKE_USD = 30000.0  # Exit when price rises $30K above entry
        
    def run_forward_simulation(self, price_path):
        """Run simulation forward through generated price path"""
        state = self.initialize_current_market_state()
        
        for day, price in enumerate(price_path):
            # Daily interest accrual
            if self.current_loan:
                self.accrue_daily_interest()
                
            # Check exit conditions (profit-taking, margin calls)
            if self.should_close_cycle(price):
                self.close_current_cycle(price, day)
                
            # Start new cycle if conditions met
            elif self.can_start_new_cycle(price):
                self.start_new_cycle(price, day)
                
        return self.generate_simulation_report()
```

---

## 📊 **DETAILED TECHNICAL SPECIFICATIONS**

### **1. Market Data Integration**
**Target File**: `new-btc-sim/leverage_war_days.py` - Replace existing Kraken API call (lines 42-66)
**Reference Files**: 
- `old-projects/leverage_sim.py` lines 89-120 (working live price fetch examples)
- `old-projects/leverage_sim2.py` lines 45-75 (fallback mechanisms)

```python
def fetch_live_btc_price():
    """Get current BTC price from multiple sources with fallback"""
    sources = [
        ('coinbase', fetch_coinbase_price),
        ('kraken', fetch_kraken_price),
        ('binance_us', fetch_binance_us_price)
    ]
    
    for source_name, fetch_func in sources:
        try:
            price = fetch_func()
            print(f"✅ Live BTC price from {source_name}: ${price:,.2f}")
            return price
        except Exception as e:
            print(f"⚠️ {source_name} failed: {e}")
    
    # Fallback to recent historical if all live sources fail
    return get_recent_historical_price()
```

### **2. Scenario-Based Price Modeling**
```python
def create_monte_carlo_price_paths(start_price, scenarios, days):
    """Create multiple realistic forward price scenarios"""
    paths = []
    
    for scenario_name, params in scenarios.items():
        for _ in range(int(1000 * params['probability'])):
            # Generate path using scenario parameters
            path = generate_gbm_path(
                start_price=start_price,
                mu=params['annual_return'] / 365,
                sigma=params['volatility'] / sqrt(365),
                days=days
            )
            paths.append({
                'scenario': scenario_name,
                'prices': path,
                'probability': params['probability']
            })
    
    return paths
```

### **3. Loan Cycle Management**
**Target File**: `new-btc-sim/leverage_war_days.py` - Replace `start_new_cycle()` method (lines 191-235)
**Contract Reference**: `new-btc-sim/Loan contract ocr.md` (Figure Lending LLC terms)
**Validation Against**: Contract parameters on page 7-8 (LTV ratios, liquidation thresholds)

```python
def start_new_cycle(self, price, day):
    """Start new loan cycle with contract-compliant sizing"""
    # First cycle: Fixed $10K loan per improvement plan
    if self.cycle_count == 0:
        loan_amount = 10000.0
        required_collateral_btc = loan_amount / (price * self.LTV_BASELINE)
    # Subsequent cycles: Progressive scaling
    else:
        collateral_btc = self.total_btc / 2.0
        loan_amount = min(
            collateral_btc * price * self.LTV_BASELINE,  # Contract limit
            collateral_btc * price * 0.4 * 0.75  # 60% crash safety limit
        )
        
    if self.total_btc >= required_collateral_btc and loan_amount >= 10000:
        self.current_loan = ContractCompliantLoan(
            principal=loan_amount,
            collateral_btc=required_collateral_btc,
            entry_price=price,
            entry_day=day
        )
        self.cycle_count += 1
        print(f"✅ Started cycle {self.cycle_count}: ${loan_amount:,.0f} loan")
```

### **4. Exit Condition Logic**
```python
def should_close_cycle(self, current_price):
    """Determine if current cycle should be closed"""
    if not self.current_loan:
        return False
        
    ltv = self.current_loan.get_current_ltv(current_price)
    profit_target = self.current_loan.entry_price + self.PROFIT_TAKE_USD
    
    # Contract-mandated liquidation
    if ltv >= self.LIQUIDATION_LTV:
        self.liquidation_reason = "Contract liquidation - 90% LTV exceeded"
        return True
        
    # Profit-taking exit
    if current_price >= profit_target:
        self.exit_reason = f"Profit target reached: ${profit_target:,.0f}"
        return True
        
    # Margin call handling
    if ltv >= self.MARGIN_CALL_LTV:
        return self.handle_margin_call(current_price)
        
    return False
```

---

## 🎲 **MONTE CARLO IMPLEMENTATION**

### **Multi-Scenario Testing**
```python
def run_comprehensive_monte_carlo(num_simulations=5000):
    """Run Monte Carlo across multiple market scenarios"""
    results = {
        'liquidation_rate': [],
        'goal_achievement_rate': [],
        'avg_cycles_completed': [],
        'final_btc_holdings': [],
        'scenario_performance': {}
    }
    
    for sim in range(num_simulations):
        # Generate forward price path
        scenario = select_weighted_scenario()
        price_path = generate_scenario_price_path(scenario)
        
        # Run simulation
        sim_result = self.run_forward_simulation(price_path)
        
        # Record results
        results['liquidation_rate'].append(sim_result['liquidated'])
        results['final_btc_holdings'].append(sim_result['final_btc'])
        
        # Track scenario-specific performance
        if scenario not in results['scenario_performance']:
            results['scenario_performance'][scenario] = []
        results['scenario_performance'][scenario].append(sim_result)
    
    return analyze_monte_carlo_results(results)
```

---

## 📈 **SUCCESS METRICS & VALIDATION**

### **Technical Validation**
- [ ] **Forward Time Flow**: All price data flows forward in time
- [ ] **Current Market Start**: Simulation begins with live market prices
- [ ] **Contract Compliance**: All parameters match Figure Lending LLC exactly
- [ ] **Multiple Cycles**: Successfully completes multiple loan cycles
- [ ] **Realistic Scenarios**: Price paths reflect historical Bitcoin behavior

### **Performance Metrics**
- [ ] **Liquidation Rate**: < 15% across 5000 simulations
- [ ] **Goal Achievement**: > 30% reach 1 BTC goal within 2 years
- [ ] **Average Cycles**: 3-8 cycles per successful simulation
- [ ] **Risk-Adjusted Return**: Positive Sharpe ratio vs holding BTC

### **Scenario Coverage**
- [ ] **Bull Market Survival**: Strategy performs well in 150%+ annual gains
- [ ] **Bear Market Resilience**: < 25% liquidation rate in -60% scenarios
- [ ] **Sideways Grinding**: Profitable in low-volatility environments
- [ ] **Crash Recovery**: Handles 50%+ drawdowns with recovery

---

## 🛠️ **IMPLEMENTATION ROADMAP**

### **Week 1: Core Framework**
**Files to Modify**:
1. **`new-btc-sim/leverage_war_days.py`** - Replace live price fetching (lines 42-66)
2. **`new-btc-sim/leverage_war_days.py`** - Implement forward price path generation (new function)
3. **`new-btc-sim/leverage_war_days.py`** - Fix loan sizing logic in `start_new_cycle()` (lines 191-235)
4. **`new-btc-sim/leverage_war_days.py`** - Fix exit conditions in `run()` method (lines 140-190)

**Reference Files for Examples**:
- `old-projects/leverage_sim.py` (working price fetch patterns)
- `old-projects/leverage_sim2.py` (Monte Carlo implementations)
- `new-btc-sim/Loan contract ocr.md` (contract compliance validation)

### **Week 2: Monte Carlo Engine**
**Files to Create/Modify**:
1. **`new-btc-sim/leverage_war_days.py`** - Add scenario-based price modeling (new `generate_forward_price_scenarios()`)
2. **`new-btc-sim/leverage_war_days.py`** - Enhance `run_monte_carlo_simulation()` method (lines 311-380)
3. **`new-btc-sim/leverage_war_days.py`** - Improve export functionality (lines 400-450)
4. **`new-btc-sim/exports/`** - Updated result visualization templates

**Reference Implementation**:
- `old-projects/leverage_sim3.py` lines 200-300 (Monte Carlo patterns)
- `exports/` directory structure examples

### **Week 3: Validation & Optimization**
**Files for Testing**:
1. **`new-btc-sim/leverage_war_days.py`** - Add validation tests against contract terms
2. **`new-btc-sim/leverage_war_days.py`** - Compare with `old-projects/leverage_sim.py` results
3. **`new-btc-sim/leverage_war_days.py`** - Grid search optimization in `run_sensitivity_analysis()`
4. **`new-btc-sim/sim-improvement-plan-3.md`** - Document final results and tutorials

**Validation Against**:
- `new-btc-sim/Loan contract ocr.md` (contract compliance)
- `old-projects/exports/` (historical baseline comparisons)
- `new-btc-sim/Loan Mermaids.md` (strategy logic validation)

---

## ⚠️ **RISK MANAGEMENT**

### **Simulation Risks**
- **Model Risk**: Monte Carlo scenarios may not capture future reality
- **Parameter Risk**: Contract terms could change
- **Market Risk**: Extreme events beyond historical precedent

### **Mitigation Strategies**
```python
def add_risk_management_features():
    """Enhanced risk management beyond basic contract terms"""
    return {
        'position_sizing': 'Never risk more than 50% of holdings as collateral',
        'stress_testing': 'Test against 90% drawdown scenarios',
        'diversification': 'Consider correlation with traditional markets',
        'exit_discipline': 'Automatic exits at predefined thresholds'
    }
```

---

## 🎯 **EXPECTED OUTCOMES**

### **Simulation Improvements in `new-btc-sim/leverage_war_days.py`**
- **Realistic Results**: Forward simulation eliminates impossible scenarios seen in current logs
- **Multiple Cycles**: 5-10 loan cycles per 2-year simulation period (vs current 1 cycle)
- **Contract Accuracy**: 100% compliance with `new-btc-sim/Loan contract ocr.md` terms
- **Scenario Coverage**: Performance across all major market conditions

### **Decision Support via `new-btc-sim/exports/` Output**
- **Clear Risk Metrics**: Precise liquidation probabilities in CSV exports
- **Parameter Optimization**: Best LTV ratios and profit targets in sensitivity analysis
- **Market Timing**: When to deploy vs hold cash based on scenario modeling
- **Capital Allocation**: Optimal position sizing strategies documented in results

### **File Deliverables**
- **Fixed Core**: `new-btc-sim/leverage_war_days.py` with working forward simulation
- **Results**: Enhanced `new-btc-sim/exports/` with realistic multi-cycle data
- **Documentation**: Updated `new-btc-sim/sim-improvement-plan-3.md` with validation results
- **Compliance**: Full validation against `new-btc-sim/Loan contract ocr.md` parameters

This pure forward simulation approach will provide realistic, actionable insights while maintaining strict contract compliance and mathematical rigor, with all changes tracked in the specified files for maximum AI agent context.
