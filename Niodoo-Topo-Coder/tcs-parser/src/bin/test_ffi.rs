// Simple test of Rust→Python FFI bridge
// This bypasses the tree-sitter C linker issue

use pyo3::prelude::*;
use pyo3::types::PyModule;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        // Add current directory to Python path
        let sys = PyModule::import_bound(py, "sys")?;
        sys.getattr("path")?
            .downcast::<pyo3::types::PyList>()?
            .insert(0, "tcs-parser")?;

        // Import test module
        let test_module = PyModule::import_bound(py, "test_bridge")?;

        // Call function
        let result: String = test_module
            .getattr("hello_from_python")?
            .call1(("Claude",))?
            .extract()?;

        println!("{}", result);
        println!("\n✓ Python FFI bridge works! Tree-sitter linker bypassed.");

        Ok(())
    })
}
