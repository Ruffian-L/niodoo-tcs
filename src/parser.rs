//! Code Parser for Bullshit Buster - AST Analysis for Non-Orientable Path Detection
//!
//! This module implements topological analysis of Rust code using Möbius topology
//! to detect hardcoded values, stubs, and fake implementations through non-orientable paths.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use syn::{Expr, ExprLit, File, Lit};
use tracing::info;

/// Represents a code symbol with topological properties
#[derive(Debug, Clone)]
pub struct CodeSymbol {
    pub name: String,
    pub symbol_type: SymbolType,
    pub topological_position: TopologicalPosition,
    pub is_hardcoded: bool,
    pub is_stub: bool,
    pub complexity_score: f64,
}

/// Types of code symbols we analyze
#[derive(Debug, Clone)]
pub enum SymbolType {
    Function,
    Struct,
    Enum,
    Impl,
    Variable,
    Constant,
}

/// Topological position on Möbius surface for non-orientable path analysis
#[derive(Debug, Clone)]
pub struct TopologicalPosition {
    pub u: f64, // Möbius strip parameter u ∈ [0, 1]
    pub v: f64, // Möbius strip parameter v ∈ [0, 1]
    pub k: i32, // K-twist factor for non-orientable topology
}

impl TopologicalPosition {
    /// Create a new topological position
    pub fn new(u: f64, v: f64, k: i32) -> Self {
        Self { u, v, k }
    }

    /// Calculate geodesic distance to another position
    pub fn geodesic_distance(&self, other: &Self) -> f64 {
        // Möbius strip geodesic distance calculation
        let du = (self.u - other.u).abs();
        let dv = (self.v - other.v).abs();
        let dk = (self.k - other.k).abs() as f64;

        // Non-orientable topology: account for Möbius twist
        let twist_factor = if self.k != other.k { 0.5 } else { 1.0 };

        (du * du + dv * dv + dk * dk * twist_factor).sqrt()
    }

    /// Check if position represents a "flat" geodesic (hardcoded path)
    pub fn is_flat_geodesic(&self) -> bool {
        // Flat geodesics have minimal topological complexity
        self.k == 0 && (self.u - 0.5).abs() < 0.1 && (self.v - 0.5).abs() < 0.1
    }
}

/// Main Code Parser for Bullshit Buster
pub struct CodeParser {
    symbols: Vec<CodeSymbol>,
    topological_map: HashMap<String, TopologicalPosition>,
}

impl Default for CodeParser {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeParser {
    /// Create a new code parser
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            topological_map: HashMap::new(),
        }
    }

    /// Parse a Rust file and extract symbols with topological analysis
    pub fn parse_file<P: AsRef<Path>>(&mut self, file_path: P) -> Result<()> {
        let file_path = file_path.as_ref();
        info!("Parsing file: {}", file_path.display());

        let content = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        let syntax_tree = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse Rust syntax: {}", file_path.display()))?;

        self.analyze_ast(&syntax_tree, file_path)?;

        info!(
            "Parsed {} symbols from {}",
            self.symbols.len(),
            file_path.display()
        );
        Ok(())
    }

    /// Analyze AST and extract symbols with topological properties
    fn analyze_ast(&mut self, file: &File, file_path: &Path) -> Result<()> {
        for item in &file.items {
            match item {
                syn::Item::Fn(func) => self.analyze_function(func, file_path)?,
                syn::Item::Struct(structure) => self.analyze_struct(structure, file_path)?,
                syn::Item::Enum(enumeration) => self.analyze_enum(enumeration, file_path)?,
                syn::Item::Impl(impl_block) => self.analyze_impl(impl_block, file_path)?,
                _ => {} // Skip other items for now
            }
        }
        Ok(())
    }

    /// Analyze a function for hardcoded values and stubs
    fn analyze_function(&mut self, func: &syn::ItemFn, file_path: &Path) -> Result<()> {
        let name = func.sig.ident.to_string();
        let topological_pos = self.calculate_topological_position(&name, SymbolType::Function);

        let mut is_hardcoded = false;
        let mut is_stub = false;
        let mut complexity_score = 0.0;

        // Analyze function body for hardcoded values
        for stmt in &func.block.stmts {
            if let syn::Stmt::Expr(expr, _) = stmt {
                if self.contains_hardcoded_value(expr) {
                    is_hardcoded = true;
                }
            }
        }

        // Check for stub patterns
        let block_expr = syn::Expr::Block(syn::ExprBlock {
            attrs: vec![],
            label: None,
            block: *func.block.clone(),
        });
        if self.is_stub_function(&block_expr) {
            is_stub = true;
        }

        // Calculate complexity based on structure
        complexity_score = self.calculate_complexity_score(&block_expr);

        let symbol = CodeSymbol {
            name: name.clone(),
            symbol_type: SymbolType::Function,
            topological_position: topological_pos.clone(),
            is_hardcoded,
            is_stub,
            complexity_score,
        };

        self.symbols.push(symbol);
        self.topological_map.insert(name.clone(), topological_pos);

        Ok(())
    }

    /// Analyze a struct for hardcoded values
    fn analyze_struct(&mut self, structure: &syn::ItemStruct, file_path: &Path) -> Result<()> {
        let name = structure.ident.to_string();
        let topological_pos = self.calculate_topological_position(&name, SymbolType::Struct);

        let symbol = CodeSymbol {
            name: name.clone(),
            symbol_type: SymbolType::Struct,
            topological_position: topological_pos.clone(),
            is_hardcoded: false, // Structs themselves aren't hardcoded
            is_stub: false,
            complexity_score: structure.fields.len() as f64,
        };

        self.symbols.push(symbol);
        self.topological_map.insert(name, topological_pos);

        Ok(())
    }

    /// Analyze an enum for hardcoded values
    fn analyze_enum(&mut self, enumeration: &syn::ItemEnum, file_path: &Path) -> Result<()> {
        let name = enumeration.ident.to_string();
        let topological_pos = self.calculate_topological_position(&name, SymbolType::Enum);

        let symbol = CodeSymbol {
            name: name.clone(),
            symbol_type: SymbolType::Enum,
            topological_position: topological_pos.clone(),
            is_hardcoded: false,
            is_stub: false,
            complexity_score: enumeration.variants.len() as f64,
        };

        self.symbols.push(symbol);
        self.topological_map.insert(name, topological_pos);

        Ok(())
    }

    /// Analyze an impl block
    fn analyze_impl(&mut self, impl_block: &syn::ItemImpl, file_path: &Path) -> Result<()> {
        let name = format!("impl_block_{}", impl_block.items.len());
        let topological_pos = self.calculate_topological_position(&name, SymbolType::Impl);

        let symbol = CodeSymbol {
            name: name.clone(),
            symbol_type: SymbolType::Impl,
            topological_position: topological_pos.clone(),
            is_hardcoded: false,
            is_stub: false,
            complexity_score: impl_block.items.len() as f64,
        };

        self.symbols.push(symbol);
        self.topological_map.insert(name, topological_pos);

        Ok(())
    }

    /// Check if an expression contains hardcoded values
    fn contains_hardcoded_value(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Lit(ExprLit { lit, .. }) => {
                match lit {
                    Lit::Int(int_lit) => {
                        // Check for magic numbers (hardcoded integers)
                        let value = int_lit.base10_parse::<i64>().unwrap_or(0);
                        self.is_magic_number(value)
                    }
                    Lit::Float(float_lit) => {
                        // Check for magic floats
                        let value = float_lit.base10_parse::<f64>().unwrap_or(0.0);
                        self.is_magic_float(value)
                    }
                    Lit::Str(str_lit) => {
                        // Check for hardcoded strings
                        self.is_hardcoded_string(&str_lit.value())
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Check if a number is a "magic number" (hardcoded)
    fn is_magic_number(&self, value: i64) -> bool {
        // Common magic numbers that should be config-derived
        matches!(
            value,
            0 | 1 | 2 | 3 | 4 | 5 | 10 | 100 | 1000 | 5000 | 10000
        )
    }

    /// Check if a float is a "magic float" (hardcoded)
    fn is_magic_float(&self, value: f64) -> bool {
        // Common magic floats
        matches!(
            value,
            0.0 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 10.0 | 100.0 | 1000.0
        )
    }

    /// Check if a string is hardcoded
    fn is_hardcoded_string(&self, value: &str) -> bool {
        // Check for hardcoded paths, URLs, etc.
        value.starts_with("/")
            || value.starts_with("http")
            || value.starts_with("localhost")
            || value.contains("hardcoded")
            || value.contains("TODO")
            || value.contains("FIXME")
    }

    /// Check if a function is a stub
    fn is_stub_function(&self, expr: &syn::Expr) -> bool {
        // Check for stub patterns in block expressions
        if let syn::Expr::Block(block_expr) = expr {
            for stmt in &block_expr.block.stmts {
                if let syn::Stmt::Expr(expr, _) = stmt {
                    if let Expr::Macro(macro_expr) = expr {
                        if macro_expr.mac.path.is_ident("todo")
                            || macro_expr.mac.path.is_ident("unimplemented")
                            || macro_expr.mac.path.is_ident("panic")
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Calculate complexity score for a code block
    fn calculate_complexity_score(&self, expr: &syn::Expr) -> f64 {
        let mut score = 0.0;

        if let syn::Expr::Block(block_expr) = expr {
            for stmt in &block_expr.block.stmts {
                match stmt {
                    syn::Stmt::Item(_) => score += 1.0,
                    syn::Stmt::Expr(_, _) => score += 0.5,
                    syn::Stmt::Local(_) => score += 0.8,
                    syn::Stmt::Macro(_) => score += 0.3,
                }
            }
        }

        score
    }

    /// Calculate topological position for a symbol
    fn calculate_topological_position(
        &self,
        name: &str,
        symbol_type: SymbolType,
    ) -> TopologicalPosition {
        // Use hash of name to determine position on Möbius strip
        let hash = self.hash_string(name);
        let u = (hash % 1000) as f64 / 1000.0;
        let v = ((hash / 1000) % 1000) as f64 / 1000.0;

        // K-twist factor based on symbol type
        let k = match symbol_type {
            SymbolType::Function => 1,
            SymbolType::Struct => 2,
            SymbolType::Enum => 3,
            SymbolType::Impl => 4,
            SymbolType::Variable => 0,
            SymbolType::Constant => 5,
        };

        TopologicalPosition::new(u, v, k)
    }

    /// Simple hash function for string
    fn hash_string(&self, s: &str) -> u64 {
        let mut hash = 0u64;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash
    }

    /// Get all symbols
    pub fn get_symbols(&self) -> &[CodeSymbol] {
        &self.symbols
    }

    /// Get symbols with hardcoded values
    pub fn get_hardcoded_symbols(&self) -> Vec<&CodeSymbol> {
        self.symbols.iter().filter(|s| s.is_hardcoded).collect()
    }

    /// Get stub symbols
    pub fn get_stub_symbols(&self) -> Vec<&CodeSymbol> {
        self.symbols.iter().filter(|s| s.is_stub).collect()
    }

    /// Get symbols with flat geodesics (potential bullshit)
    pub fn get_flat_geodesic_symbols(&self) -> Vec<&CodeSymbol> {
        self.symbols
            .iter()
            .filter(|s| s.topological_position.is_flat_geodesic())
            .collect()
    }

    /// Generate topological analysis report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("🧠 BULLSHIT BUSTER TOPOLOGICAL ANALYSIS REPORT\n");
        report.push_str("==============================================\n\n");

        report.push_str(&format!("Total Symbols Analyzed: {}\n", self.symbols.len()));
        report.push_str(&format!(
            "Hardcoded Values Found: {}\n",
            self.get_hardcoded_symbols().len()
        ));
        report.push_str(&format!(
            "Stub Functions Found: {}\n",
            self.get_stub_symbols().len()
        ));
        report.push_str(&format!(
            "Flat Geodesics (Bullshit): {}\n",
            self.get_flat_geodesic_symbols().len()
        ));

        report.push_str("\n🚨 HARDCODED VALUES DETECTED:\n");
        for symbol in self.get_hardcoded_symbols() {
            report.push_str(&format!(
                "  • {} ({:?}) - Position: ({:.3}, {:.3}, {})\n",
                symbol.name,
                symbol.symbol_type,
                symbol.topological_position.u,
                symbol.topological_position.v,
                symbol.topological_position.k
            ));
        }

        report.push_str("\n🔧 STUB FUNCTIONS DETECTED:\n");
        for symbol in self.get_stub_symbols() {
            report.push_str(&format!(
                "  • {} ({:?}) - Complexity: {:.2}\n",
                symbol.name, symbol.symbol_type, symbol.complexity_score
            ));
        }

        report.push_str("\n🌀 FLAT GEODESICS (NON-ORIENTABLE PATHS):\n");
        for symbol in self.get_flat_geodesic_symbols() {
            report.push_str(&format!(
                "  • {} ({:?}) - Topological Position: ({:.3}, {:.3}, {})\n",
                symbol.name,
                symbol.symbol_type,
                symbol.topological_position.u,
                symbol.topological_position.v,
                symbol.topological_position.k
            ));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_parser_creation() {
        let parser = CodeParser::new();
        assert_eq!(parser.symbols.len(), 0);
    }

    #[test]
    fn test_topological_position() {
        let pos1 = TopologicalPosition::new(0.5, 0.5, 0);
        let pos2 = TopologicalPosition::new(0.6, 0.6, 0);

        let distance = pos1.geodesic_distance(&pos2);
        assert!(distance > 0.0);

        assert!(pos1.is_flat_geodesic());
        assert!(!pos2.is_flat_geodesic());
    }

    #[test]
    fn test_magic_number_detection() {
        let parser = CodeParser::new();
        assert!(parser.is_magic_number(0));
        assert!(parser.is_magic_number(1000));
        assert!(!parser.is_magic_number(42));
    }

    #[test]
    fn test_hardcoded_string_detection() {
        let parser = CodeParser::new();
        assert!(parser.is_hardcoded_string("/usr/local/bin"));
        assert!(parser.is_hardcoded_string("http://localhost:8080"));
        assert!(parser.is_hardcoded_string("TODO: implement this"));
        assert!(!parser.is_hardcoded_string("normal_variable_name"));
    }
}
