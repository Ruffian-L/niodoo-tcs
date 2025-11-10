// Complex example with loops, matches, and nested control flow
fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn sum_evens(max: u32) -> u32 {
    let mut sum = 0;
    for i in 0..max {
        if i % 2 == 0 {
            sum += i;
        }
    }
    sum
}

fn main() {
    println!("Fibonacci(10) = {}", fibonacci(10));
    println!("Sum of evens to 100 = {}", sum_evens(100));
}
