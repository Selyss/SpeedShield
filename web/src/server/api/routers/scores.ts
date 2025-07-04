import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { scores } from "~/server/db/schema";
import { sql, count, and, gte, lte, inArray, desc } from "drizzle-orm";

export const scoresRouter = createTRPCRouter({
  getScores: publicProcedure
    .input(z.object({
      limit: z.number().min(1).max(10000).default(500),
      offset: z.number().min(0).default(0),
      bounds: z.object({
        north: z.number(),
        south: z.number(),
        east: z.number(),
        west: z.number(),
      }).optional(),
      riskCategories: z.array(z.enum(["Low", "Medium", "High", "Very High", "Critical"])).optional(),
      minScore: z.number().min(0).max(1).optional(),
      maxScore: z.number().min(0).max(1).optional(),
      loadAll: z.boolean().default(false), // New parameter to load all data in chunks
    }))
    .query(async ({ ctx, input }) => {
      const conditions = [];

      // Add geographic bounds filtering
      if (input.bounds) {
        conditions.push(
          gte(scores.latitude, input.bounds.south),
          lte(scores.latitude, input.bounds.north),
          gte(scores.longitude, input.bounds.west),
          lte(scores.longitude, input.bounds.east)
        );
      }

      // Add risk category filtering
      if (input.riskCategories && input.riskCategories.length > 0) {
        conditions.push(inArray(scores.risk_category, input.riskCategories));
      }

      // Add score range filtering
      if (input.minScore !== undefined) {
        conditions.push(gte(scores.final_score, input.minScore));
      }
      if (input.maxScore !== undefined) {
        conditions.push(lte(scores.final_score, input.maxScore));
      }

      const whereConditions = conditions.length > 0 ? and(...conditions) : undefined;

      // When fetching for the map, we want a more balanced result set
      // that doesn't just show the highest risk scores.
      const orderByClause = input.bounds
        ? sql`RANDOM()`
        : desc(scores.final_score);

      const [data, totalCountResult] = await Promise.all([
        ctx.db
          .select({
            latitude: scores.latitude,
            longitude: scores.longitude,
            risk_category: scores.risk_category,
          })
          .from(scores)
          .where(whereConditions)
          .orderBy(orderByClause)
          .limit(input.limit)
          .offset(input.offset),
        ctx.db
          .select({ count: count() })
          .from(scores)
          .where(whereConditions)
      ]);

      const totalCount = totalCountResult[0]?.count ?? 0;

      return {
        data,
        pagination: {
          total: totalCount,
          limit: input.limit,
          offset: input.offset,
          hasMore: input.offset + input.limit < totalCount,
          page: Math.floor(input.offset / input.limit) + 1,
          totalPages: Math.ceil(totalCount / input.limit),
        },
        filters: {
          bounds: input.bounds,
          riskCategories: input.riskCategories,
          minScore: input.minScore,
          maxScore: input.maxScore,
        },
      };
    }),

  // New endpoint specifically for map rendering that ensures balanced representation
  getScoresForMap: publicProcedure
    .input(z.object({
      bounds: z.object({
        north: z.number(),
        south: z.number(),
        east: z.number(),
        west: z.number(),
      }),
      riskCategories: z.array(z.enum(["Low", "Medium", "High", "Very High", "Critical"])).optional(),
      maxPerCategory: z.number().min(1).max(500).default(200), // Max points per risk category
    }))
    .query(async ({ ctx, input }) => {
      const conditions = [
        gte(scores.latitude, input.bounds.south),
        lte(scores.latitude, input.bounds.north),
        gte(scores.longitude, input.bounds.west),
        lte(scores.longitude, input.bounds.east)
      ];

      // Add risk category filtering
      if (input.riskCategories && input.riskCategories.length > 0) {
        conditions.push(inArray(scores.risk_category, input.riskCategories));
      }

      const whereConditions = and(...conditions);      // Get balanced sample from each risk category
      const categoryQueries = (input.riskCategories ?? ["Low", "Medium", "High", "Very High", "Critical"]).map(category => 
        ctx.db
          .select()
          .from(scores)
          .where(and(whereConditions, sql`${scores.risk_category} = ${category}`))
          .orderBy(sql`RANDOM()`) // Random sampling for visual distribution
          .limit(input.maxPerCategory)
      );

      const results = await Promise.all(categoryQueries);
      const data = results.flat();

      // Get total count for statistics
      const totalCountResult = await ctx.db
        .select({ count: count() })
        .from(scores)
        .where(whereConditions);

      return {
        data,
        total: totalCountResult[0]?.count ?? 0,
        sampledCount: data.length,
        bounds: input.bounds,
      };
    }),
  getHighRiskScores: publicProcedure
    .input(z.object({
      limit: z.number().min(1).max(500).default(50),
      minScore: z.number().default(0.7),
      bounds: z.object({
        north: z.number(),
        south: z.number(),
        east: z.number(),
        west: z.number(),
      }).optional(),
    }))
    .query(async ({ ctx, input }) => {
      const conditions = [gte(scores.final_score, input.minScore)];

      if (input.bounds) {
        conditions.push(
          gte(scores.latitude, input.bounds.south),
          lte(scores.latitude, input.bounds.north),
          gte(scores.longitude, input.bounds.west),
          lte(scores.longitude, input.bounds.east)
        );
      }

      return ctx.db
        .select()
        .from(scores)
        .where(and(...conditions))
        .orderBy(desc(scores.final_score))
        .limit(input.limit);
    }),

  getRiskCategoryStats: publicProcedure
    .input(z.object({
      bounds: z.object({
        north: z.number(),
        south: z.number(),
        east: z.number(),
        west: z.number(),
      }).optional(),
    }))
    .query(async ({ ctx, input }) => {
      const conditions = [];

      if (input.bounds) {
        conditions.push(
          gte(scores.latitude, input.bounds.south),
          lte(scores.latitude, input.bounds.north),
          gte(scores.longitude, input.bounds.west),
          lte(scores.longitude, input.bounds.east)
        );
      }

      const whereConditions = conditions.length > 0 ? and(...conditions) : undefined;

      const stats = await ctx.db
        .select({
          risk_category: scores.risk_category,
          count: count(),
          avg_score: sql<number>`AVG(${scores.final_score})`,
          min_score: sql<number>`MIN(${scores.final_score})`,
          max_score: sql<number>`MAX(${scores.final_score})`,
        })
        .from(scores)
        .where(whereConditions)
        .groupBy(scores.risk_category)
        .orderBy(sql`
          CASE ${scores.risk_category}
            WHEN 'Critical' THEN 1
            WHEN 'Very High' THEN 2
            WHEN 'High' THEN 3
            WHEN 'Medium' THEN 4
            WHEN 'Low' THEN 5
            ELSE 6
          END
        `);

      const totalCount = await ctx.db
        .select({ count: count() })
        .from(scores)
        .where(whereConditions);

      return {
        stats,
        total: totalCount[0]?.count ?? 0,
      };
    }),

  getScoreDetails: publicProcedure
    .input(z.object({
      latitude: z.number(),
      longitude: z.number(),
    }))
    .query(async ({ ctx, input }) => {
      const score = await ctx.db
        .select()
        .from(scores)
        .where(and(
          sql`${scores.latitude} = ${input.latitude}`,
          sql`${scores.longitude} = ${input.longitude}`
        ))
        .limit(1);

      if (score.length === 0) {
        throw new Error("Score not found for the given coordinates");
      }

      return score[0];
    }),
});
