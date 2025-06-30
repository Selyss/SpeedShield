import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { scores } from "~/server/db/schema";
import { sql, count, and, gte, lte } from "drizzle-orm";

export const scoresRouter = createTRPCRouter({
  getScores: publicProcedure
    .input(z.object({
      limit: z.number().min(1).max(1000).default(100),
      offset: z.number().min(0).default(0),
      bounds: z.object({
        north: z.number(),
        south: z.number(),
        east: z.number(),
        west: z.number(),
      }).optional(),
    }))
    .query(async ({ ctx, input }) => {
      const whereConditions = input.bounds
        ? and(
            gte(scores.latitude, input.bounds.south),
            lte(scores.latitude, input.bounds.north),
            gte(scores.longitude, input.bounds.west),
            lte(scores.longitude, input.bounds.east)
          )
        : undefined;

      const [data, totalCountResult] = await Promise.all([
        ctx.db
          .select()
          .from(scores)
          .where(whereConditions)
          .limit(input.limit)
          .offset(input.offset)
          .orderBy(scores.final_score),
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
        },
      };
    }),

  getHighRiskScores: publicProcedure
    .input(z.object({
      limit: z.number().min(1).max(500).default(50),
      minScore: z.number().default(0.7),
    }))
    .query(async ({ ctx, input }) => {
      return ctx.db
        .select()
        .from(scores)
        .where(gte(scores.final_score, input.minScore))
        .orderBy(sql`${scores.final_score} DESC`)
        .limit(input.limit);
    }),
});
