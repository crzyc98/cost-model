---
trigger: always_on
---

# AI Assistant Rules for Cost Model Project

## Purpose
To assist in the development, debugging, enhancement, and accurate documentation of the workforce cost model project. This includes orchestrating and simulating workforce projections, snapshot generation, event logging, employee terminations, hires, promotions, and compensation adjustments. The assistant will help maintain clear, structured, and error-free Python code aligned with data analytics best practices and strategic business objectives.

## Tone
Maintain a highly analytical, structured, precise, patient, and professional tone. Communication should be clear, insightful, and respectful, reflecting deep analytical understanding, attention to detail, and pragmatism consistent with a senior-level data science and analytics professional.

## Project Structure
- **Core Simulation Logic**: `cost_model/simulation.py`
- **State Management**: `cost_model/state/`
  - `snapshot.py` - Snapshot creation and management
  - `snapshot_update.py` - Snapshot update logic
  - `event_log.py` - Event logging functionality
  - `schema.py` - Data schema definitions
- **Dynamics**: `cost_model/dynamics/`
  - `engine.py` - Core dynamics orchestration
  - `hiring.py` - Hiring logic
  - `termination.py` - Termination logic
  - `compensation.py` - Compensation adjustments
- **Plan Rules**: `cost_model/plan_rules/` - Business rules for benefits and contributions
- **Projections**: `cost_model/projections/`
  - `hazard.py` - Hazard modeling
  - `cli.py` - Command line interface
- **Data**: `cost_model/data/` - Data loading and preprocessing
- **ML**: `cost_model/ml/` - Machine learning models
- **Reporting**: `cost_model/reporting/` - Reporting utilities

## Coding Standards
1. **Type Hints**: Use Python type hints consistently
2. **Logging**: Use the configured logging system instead of print statements
3. **Error Handling**: Implement proper error handling with meaningful messages
4. **Documentation**: Maintain clear docstrings and comments
5. **Testing**: Ensure new code has corresponding tests in the `tests/` directory

## Key Conventions
1. **Snapshot Management**:
   - Use `snapshot.py` for snapshot creation
   - Update snapshots using functions in `snapshot_update.py`
   - Maintain data consistency with schema definitions in `schema.py`

2. **Event Logging**:
   - Log all significant state changes
   - Use structured logging for better querying
   - Include relevant context in log messages

3. **Configuration**:
   - Store configuration in YAML files under `config/`
   - Avoid hardcoding parameters in the code
   - Use configuration namespaces for different environments

4. **Performance**:
   - Optimize for large datasets
   - Use vectorized operations with pandas/numpy
   - Be mindful of memory usage with large snapshots

## Common Tasks
1. **Adding New Features**:
   - Follow the existing project structure
   - Add appropriate type hints and docstrings
   - Include unit tests
   - Update documentation

2. **Debugging**:
   - Check logs in `output_dev/projection_logs/`
   - Use structured logging for better traceability
   - Validate data against schema definitions

3. **Testing**:
   - Run tests with `pytest`
   - Add tests for new functionality
   - Maintain test coverage

## Best Practices
1. **Code Organization**:
   - Keep files focused and modular
   - Group related functionality together
   - Follow the single responsibility principle

2. **Data Handling**:
   - Validate input data
   - Handle missing values appropriately
   - Use pandas efficiently

3. **Documentation**:
   - Document public APIs
   - Include examples in docstrings
   - Keep README files up to date

## Common Pitfalls
1. **Memory Issues**:
   - Be cautious with large DataFrames
   - Use chunking for large operations
   - Clear unused variables

2. **Performance Bottlenecks**:
   - Avoid loops over DataFrame rows
   - Use vectorized operations
   - Profile code to identify slow sections

3. **Data Consistency**:
   - Validate data after transformations
   - Check for index uniqueness
   - Handle edge cases

## Top Rules
1. **Read and Understand Everything**
   - Read every provided document, instruction, or code snippet line by line
   - Never summarize or skim important details

2. **Mandatory Snapshot Reference**
   - Always refer to the latest project documentation and codebase
   - Ensure understanding of the current state before making changes

3. **Structured Logging**
   - Use the configured logging system consistently
   - Include relevant context in log messages
   - Log at appropriate levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

4. **Code Quality**
   - Follow PEP 8 style guide
   - Use type hints consistently
   - Write meaningful docstrings and comments

5. **Testing**
   - Write tests for new functionality
   - Maintain test coverage
   - Run tests before submitting changes

## Getting Help
1. Check the project documentation
2. Review existing code for similar patterns
3. Use the logging system to trace issues
4. Consult the project's issue tracker for known problems

To have a robust, fully functional, and accurate workforce cost simulation model capable of:
	•	Running deterministic and stochastic workforce projections over multiple years.
	•	Accurately modeling hiring, terminations (experienced and new-hire), promotions, compensation adjustments, and other workforce-related events.
	•	Producing clear, structured, and accurate data snapshots and comprehensive event logs.
	•	Ensuring consistency, accuracy, and transparency across all modules and code, with rigorous validation and zero tolerance for bugs or inaccuracies.
	•	Clearly documented and easily maintainable codebase with automated logging and tracking mechanisms.