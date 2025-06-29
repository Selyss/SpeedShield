import MapPage from "~/components/MapPage";
import { api } from "~/trpc/server";

export default async function Home() {
  return <MapPage />;
}
