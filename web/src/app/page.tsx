import MapPage from "~/components/MapPage";
import { api } from "~/trpc/server";

export default async function Home() {
  const hello = await api.hello.getAll();
  // console.log(hello.message); // Log the message from the server
  return <MapPage />;
}
