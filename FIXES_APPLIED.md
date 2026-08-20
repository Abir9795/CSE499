# Fixes Applied - Reinforcement Loop Issues

## Problem Summary
The system was stopping prematurely after the first failed attempt with "MODEL REPEATED AN EARLIER SOLUTION" error, preventing refinement attempts from continuing.

## Root Causes Identified

### 1. **Critical: Hardcoded Fallback Response in LLMClient** ⚠️
**File**: `skillforge/src/agents/llm_client.py`

**Issue**: The `_fallback_response()` method was returning hardcoded answers for a maximum subarray problem, regardless of the actual problem being solved. This caused:
- Wrong test cases to be generated for any problem
- The system to think every problem was a "maximum subarray" problem
- Consistent failures and model confusion

**Fix**: Replaced fallback responses with proper error raising. Now when the LLM service is unavailable, the system will clearly report the error instead of silently corrupting results.

```python
# Before: Returned hardcoded maximum subarray solution
def _fallback_response(self, prompt, system=None):
    if system and "problem analyzer" in system.lower():
        return '{"problem_type": "array", "constraints": "n >= 1", ...}'
    
# After: Raises clear error
def _fallback_response(self, prompt, system=None):
    if ollama is None:
        raise RuntimeError(f"ollama module is not available...")
```

### 2. **Repeated Candidate Detection Logic**
**File**: `skillforge/src/reinforcement_loop.py`

**Issue**: The loop immediately stopped if the first refinement attempt returned the same code, treating it as "giving up" rather than "needs another try".

**Fix**: Modified to allow up to 2 consecutive repeated candidates before stopping:
- Track consecutive repeats instead of just checking if seen before
- Reset counter when a new unique candidate is generated
- Continue trying unless we hit the max_repeats threshold

```python
# Before: Stopped immediately on any repeat
if fingerprint in seen_candidates:
    result.stop_reason = "repeated_candidate"
    break

# After: Allow multiple repeats
repeated_count += 1
if repeated_count >= max_repeats:
    result.stop_reason = "repeated_candidate"
    break
```

### 3. **Refinement Agent Prompt Improvement**
**File**: `skillforge/src/agents/refinement_agent.py`

**Issue**: The refinement prompt didn't explicitly tell the LLM to generate a different solution, which sometimes led to identical code being returned.

**Fix**: 
- Added explicit instruction: "The solution must be different from the previous one"
- Increased temperature slightly (0.2 → 0.3 max) to encourage diversity
- Added guidance to try a different approach if needed

### 4. **Task Spec Parser Enhancement**
**File**: `skillforge/src/agents/task_spec.py`

**Issue**: The prompt for parsing problem statements was too vague and the LLM could confuse single integers with arrays.

**Fix**: Enhanced system prompt with critical instructions:
- Explicitly distinguish between single integer math problems and array problems
- Added note to extract examples from problem statement only (don't make them up)
- Better formatting to reduce ambiguity

## Testing

### Before Fixes:
```
ATTEMPT 1: Generated wrong problem (maximum subarray instead of cube)
ATTEMPT 2: MODEL REPEATED AN EARLIER SOLUTION (Stopped)
```

### After Fixes:
- System now properly validates LLM responses
- Raises clear errors when LLM service is unavailable
- Allows refinement attempts to continue even if early attempts repeat
- Better prompts help LLM understand problem statements correctly

## Installation Note

If you see: `RuntimeError: ollama module is not available`

You need to install ollama:
```bash
pip install ollama
```

Or ensure the ollama service is running locally on your machine.

## Files Modified

1. `skillforge/src/agents/llm_client.py` - Fixed fallback response
2. `skillforge/src/reinforcement_loop.py` - Improved repeat detection logic
3. `skillforge/src/agents/refinement_agent.py` - Better prompts for diversity
4. `skillforge/src/agents/task_spec.py` - Clearer problem parsing instructions
5. `skillforge/run_reinforcement.py` - Removed debug output

## Impact

- ✅ System no longer silently fails with wrong problem definitions
- ✅ Refinement loop continues to attempt improvements instead of giving up early
- ✅ Better error messages when LLM service is unavailable
- ✅ More reliable problem understanding
