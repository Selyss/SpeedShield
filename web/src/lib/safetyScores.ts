export const RISK_CATEGORIES = [
  "Low",
  "Medium", 
  "High",
  "Very High",
  "Critical"
] as const;

export type RiskCategory = typeof RISK_CATEGORIES[number];

export const RISK_CATEGORY_COLORS: Record<RiskCategory, string> = {
  "Low": "#22c55e",      // green-500
  "Medium": "#eab308",   // yellow-500
  "High": "#f97316",     // orange-500
  "Very High": "#dc2626", // red-600
  "Critical": "#7c2d12"   // red-900
};

export const RISK_CATEGORY_ORDER: Record<RiskCategory, number> = {
  "Low": 1,
  "Medium": 2,
  "High": 3,
  "Very High": 4,
  "Critical": 5
};

export function getRiskCategoryColor(category: string): string {
  return RISK_CATEGORY_COLORS[category as RiskCategory] ?? "#6b7280"; // gray-500 as fallback
}

export function shouldShowRiskCategory(category: string, selectedCategories: RiskCategory[]): boolean {
  return selectedCategories.includes(category as RiskCategory);
}

// Helper to filter categories by minimum risk level
export function getMinimumRiskCategories(minCategory: RiskCategory): RiskCategory[] {
  const minOrder = RISK_CATEGORY_ORDER[minCategory];
  return RISK_CATEGORIES.filter(cat => RISK_CATEGORY_ORDER[cat] >= minOrder);
}
