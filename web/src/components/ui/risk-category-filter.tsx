"use client";

import * as React from "react";
import { Check } from "lucide-react";
import { Button } from "~/components/ui/button";
import {
  RISK_CATEGORIES,
  RISK_CATEGORY_COLORS,
  type RiskCategory,
} from "~/lib/safetyScores";

interface RiskCategoryFilterProps {
  selectedCategories: RiskCategory[];
  onCategoriesChange: (categories: RiskCategory[]) => void;
}

export function RiskCategoryFilter({
  selectedCategories,
  onCategoriesChange,
}: RiskCategoryFilterProps) {
  const toggleCategory = (category: RiskCategory) => {
    if (selectedCategories.includes(category)) {
      onCategoriesChange(selectedCategories.filter((c) => c !== category));
    } else {
      onCategoriesChange([...selectedCategories, category]);
    }
  };

  const selectAll = () => {
    onCategoriesChange([...RISK_CATEGORIES]);
  };

  const clearAll = () => {
    onCategoriesChange([]);
  };

  return (
    <div className="space-y-3 rounded-lg border bg-background p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Risk Categories</h3>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={selectAll}
            className="h-6 px-2 text-xs"
          >
            All
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={clearAll}
            className="h-6 px-2 text-xs"
          >
            None
          </Button>
        </div>
      </div>
      <div className="space-y-2">
        {RISK_CATEGORIES.map((category) => (
          <div
            key={category}
            className="flex items-center space-x-2 cursor-pointer"
            onClick={() => toggleCategory(category)}
          >
            <div
              className={`flex h-4 w-4 items-center justify-center rounded border ${
                selectedCategories.includes(category)
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-muted-foreground"
              }`}
            >
              {selectedCategories.includes(category) && (
                <Check className="h-3 w-3" />
              )}
            </div>
            <div className="flex items-center space-x-2">
              <div
                className="h-3 w-3 rounded-full"
                style={{ backgroundColor: RISK_CATEGORY_COLORS[category] }}
              />
              <span className="text-sm">{category}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
