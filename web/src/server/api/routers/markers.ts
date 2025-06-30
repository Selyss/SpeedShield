import { z } from "zod";
import { eq } from "drizzle-orm";

import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { markers } from "~/server/db/schema";
import { get } from "http";

export const markersRouter = createTRPCRouter({
  // Get all markers
  getAll: publicProcedure.query(({ ctx }) => {
    return ctx.db.select().from(markers);
  }),

  // Get marker by ID
  getById: publicProcedure
    .input(z.object({ id: z.number() }))
    .query(({ ctx, input }) => {
      return ctx.db.select().from(markers).where(eq(markers.id, input.id));
    }),
  getByType: publicProcedure
    .input(z.object({ markerType: z.string() }))
    .query(({ ctx, input }) => {
      return ctx.db
        .select()
        .from(markers)
        .where(eq(markers.markerType, input.markerType));
    }),
});