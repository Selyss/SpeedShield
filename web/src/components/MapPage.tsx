"use client";

import dynamic from "next/dynamic";
import { api } from "~/trpc/react";

// Dynamically import the map component to avoid SSR issues
const InteractiveMap = dynamic(() => import("~/components/InteractiveMap"), {
	ssr: false,
	loading: () => (
		<div className="flex h-screen w-full items-center justify-center bg-gray-100">
		<div className="text-lg">Loading map...</div>
		</div>
	),
});

export default function MapPage() {
	const [hello] = api.hello.hello.useSuspenseQuery({ text: "aaron" });
	
	return <>
	<p>hi {hello.greeting}</p>
	<InteractiveMap />
	</>;
}
