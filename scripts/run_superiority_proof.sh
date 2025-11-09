#!/bin/bash
# Superiority Proof Report Generator
# Aggregates ablation and A/B test results to prove system superiority

set -e

OUTPUT_DIR="${OUTPUT_DIR:-superiority_proofs}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/superiority_proof_${TIMESTAMP}.json"

mkdir -p "${OUTPUT_DIR}"

echo "🔬 Generating Superiority Proof Report..."
echo "=========================================="

# Collect ablation results
ABLATION_RESULTS_DIR="${ABLATION_RESULTS_DIR:-ablation_results}"
AB_TEST_RESULTS_DIR="${AB_TEST_RESULTS_DIR:-ab_test_results}"

# Aggregate ablation results
ABLATION_SUMMARY="{}"
if [ -d "${ABLATION_RESULTS_DIR}" ] && [ "$(ls -A ${ABLATION_RESULTS_DIR}/*.json 2>/dev/null)" ]; then
    echo "📊 Aggregating ablation results..."
    ABLATION_SUMMARY=$(cat ${ABLATION_RESULTS_DIR}/*.json | jq -s '{
        total_experiments: length,
        experiments: .,
        critical_components: [.[] | select(.comparison.regression_detected == true) | .experiment],
        component_scores: [.[] | {component: .experiment, score: .comparison.component_contribution_score}],
        average_p_value: ([.[] | .comparison.p_value] | add / length),
        average_cohens_d: ([.[] | .comparison.cohens_d_latency] | add / length)
    }')
fi

# Aggregate A/B test results
AB_TEST_SUMMARY="{}"
if [ -d "${AB_TEST_RESULTS_DIR}" ] && [ "$(ls -A ${AB_TEST_RESULTS_DIR}/*.json 2>/dev/null)" ]; then
    echo "📊 Aggregating A/B test results..."
    AB_TEST_SUMMARY=$(cat ${AB_TEST_RESULTS_DIR}/*.json | jq -s '{
        total_tests: length,
        tests: .,
        winners: [.[] | .winner],
        significant_improvements: [.[] | select(.statistical_significance == true and .comparison.latency_difference_ms < 0)],
        average_latency_improvement: ([.[] | .comparison.latency_difference_pct] | add / length),
        average_throughput_improvement: ([.[] | .comparison.throughput_difference_pct] | add / length)
    }')
fi

# Generate comprehensive report
FINAL_REPORT=$(jq -n \
    --argjson ablation "${ABLATION_SUMMARY}" \
    --argjson ab_test "${AB_TEST_SUMMARY}" \
    --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{
        timestamp: $timestamp,
        summary: {
            ablation_studies: $ablation,
            ab_tests: $ab_test
        },
        superiority_claims: {
            critical_components: ($ablation.critical_components // []),
            performance_improvements: ($ab_test.significant_improvements // []),
            component_contributions: ($ablation.component_scores // [])
        },
        recommendations: {
            essential_components: ($ablation.critical_components // []),
            optimization_opportunities: ($ab_test.significant_improvements // [] | map(.treatment_name))
        }
    }')

echo "${FINAL_REPORT}" | jq '.' > "${REPORT_FILE}"

echo ""
echo "✅ Superiority proof report generated: ${REPORT_FILE}"
echo ""
echo "📋 Summary:"
echo "   Ablation Experiments: $(echo "${FINAL_REPORT}" | jq '.summary.ablation_studies.total_experiments // 0')"
echo "   A/B Tests: $(echo "${FINAL_REPORT}" | jq '.summary.ab_tests.total_tests // 0')"
echo "   Critical Components: $(echo "${FINAL_REPORT}" | jq '.superiority_claims.critical_components | length')"
echo ""
echo "💡 Key Findings:"
echo "${FINAL_REPORT}" | jq -r '.superiority_claims.critical_components[]? | "   - \(.) is CRITICAL"'
echo "${FINAL_REPORT}" | jq -r '.recommendations.essential_components[]? | "   - \(.) is ESSENTIAL"'






