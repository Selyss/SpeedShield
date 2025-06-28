import { z } from "zod";
import { eq } from "drizzle-orm";

import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { markers } from "~/server/db/schema";

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

  // Create a new marker
  create: publicProcedure
    .input(
      z.object({
        name: z.string().min(1).max(256),
        latitude: z.number(),
        longitude: z.number(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      return ctx.db
        .insert(markers)
        .values({
          name: input.name,
          latitude: input.latitude,
          longitude: input.longitude,
        })
        .returning();
    }),

  // Update a marker
  update: publicProcedure
    .input(
      z.object({
        id: z.number(),
        name: z.string().min(1).max(256).optional(),
        latitude: z.number().optional(),
        longitude: z.number().optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const { id, ...updateData } = input;
      return ctx.db
        .update(markers)
        .set(updateData)
        .where(eq(markers.id, id))
        .returning();
    }),

  // Delete a marker
  delete: publicProcedure
    .input(z.object({ id: z.number() }))
    .mutation(async ({ ctx, input }) => {
      return ctx.db
        .delete(markers)
        .where(eq(markers.id, input.id))
        .returning();
    }),
});