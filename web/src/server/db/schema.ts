// Example model schema from the Drizzle docs
// https://orm.drizzle.team/docs/sql-schema-declaration

import { sql } from "drizzle-orm";
import { index, pgTableCreator, primaryKey } from "drizzle-orm/pg-core";
import type { Infer } from "next/dist/compiled/superstruct";

/**
 * This is an example of how to use the multi-project schema feature of Drizzle ORM. Use the same
 * database instance for multiple projects.
 *
 * @see https://orm.drizzle.team/docs/goodies#multi-project-schema
 */
export const createTable = pgTableCreator((name) => `speedshield_${name}`);

// export const posts = createTable(
//   "post",
//   (d) => ({
//     id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
//     name: d.varchar({ length: 256 }),
//     createdAt: d
//       .timestamp({ withTimezone: true })
//       .default(sql`CURRENT_TIMESTAMP`)
//       .notNull(),
//     updatedAt: d.timestamp({ withTimezone: true }).$onUpdate(() => new Date()),
//   }),
//   (t) => [index("name_idx").on(t.name)]
// );
export const markers = createTable(
  "marker",
  (d) => ({
    id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
    name: d.varchar({ length: 256 }),
    latitude: d.doublePrecision().notNull(),
    longitude: d.doublePrecision().notNull(),
    markerType: d.varchar({ length: 64 }).notNull().default("default"),
  })
)

export const scores = createTable(
  "scores",
  (d) => ({
    longitude: d.doublePrecision().notNull(),
    latitude: d.doublePrecision().notNull(),
    latest_count_id: d.bigint({mode:"bigint"}),
    predicted_risk: d.doublePrecision(),
    risk_percentile: d.doublePrecision(),
    risk_category: d.text(),
    collision_count: d.bigint({mode:"number"}),
    speed_risk: d.doublePrecision(),
    volume_risk: d.doublePrecision(),
    collision_history: d.doublePrecision(),
    heavy_share: d.doublePrecision(),
    near_school: d.boolean(),
    volume_risk_percentile: d.doublePrecision(),
    collision_history_percentile: d.doublePrecision(),
    volume_risk_category: d.text(),
    collision_history_category: d.text(),
    veh_km: d.doublePrecision(),
    in_school_zone: d.boolean(),
    near_retirement_home: d.boolean(),
    avg_daily_vol: d.doublePrecision(),
    avg_speed: d.doublePrecision(),
    avg_85th_percentile_speed: d.doublePrecision(),
    avg_95th_percentile_speed: d.doublePrecision(),
    avg_heavy_pct: d.doublePrecision(),
    has_camera: d.boolean(),
    school_risk_factor: d.integer(),
    vulnerable_population_risk: d.real(),
    camera_score: d.doublePrecision(),
    final_score: d.doublePrecision(),
    speed_risk_percentile: d.doublePrecision(),
    heavy_share_percentile: d.doublePrecision(),
    speed_risk_category: d.text(),
    heavy_share_category: d.text(),
  }),
  (t) => [
    // Composite primary key on longitude, latitude
    primaryKey({ columns: [t.longitude, t.latitude] })
  ]
);

export type Marker = typeof markers.$inferSelect;
export type Score = typeof scores.$inferSelect;

