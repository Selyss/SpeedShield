"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Button } from "~/components/ui/button";
import { api } from "~/trpc/react";
import L from "leaflet";
import { Circle, LayersControl } from "react-leaflet";

interface DialogData
  extends Record<string, string | number | boolean | Date | null | undefined> {
  title?: string;
}

interface Marker {
  id: number;
  name: string | null;
  latitude: number;
  longitude: number;
  createdAt: Date;
  updatedAt: Date | null;
}

const markerIcon = L.divIcon({
  html: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-map-pin-icon lucide-map-pin"><path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0"/><circle cx="12" cy="10" r="3"/></svg>`,
  iconAnchor: [12, 24], // Center the icon at the bottom
  className: "marker-icon", // Custom class for styling
});

// Create a separate component for the map content that will be dynamically loaded
const MapContent = dynamic(
  () =>
    import("react-leaflet").then((mod) => {
      const { MapContainer, TileLayer, useMapEvents, Marker } = mod;
      function MapClickHandler({
        onMapClick,
      }: {
        onMapClick: (data: DialogData) => void;
      }) {
        useMapEvents({
          click: (e) => {
            onMapClick({
              title: "Coordinates",
              lat: e.latlng.lat,
              lng: e.latlng.lng,
            });
          },
        });
        return null;
      }

      return function Map({
        dialogHandler,
        markers,
      }: {
        dialogHandler: (data: DialogData) => void;
        markers: Marker[];
      }) {
        return (
          <MapContainer
            center={[43.6532, -79.3832]} // Default to toronto
            zoom={13}
            style={{ height: "100%", width: "100%" }}
            className="z-0"
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <LayersControl position="topright">
              <LayersControl.Overlay name="Schools">
                <Circle
                  center={[43.6532, -79.3832]} // example school
                  radius={500} // 500m radius
                  pathOptions={{ color: "blue", fillColor: "blue", fillOpacity: 0.1 }}
                />
              </LayersControl.Overlay>
              <LayersControl.Overlay checked name="Markers">
                <>
                  {markers.map((marker) => (
                    <Marker
                      key={marker.id}
                      position={[marker.latitude, marker.longitude]}
                      icon={markerIcon}
                      eventHandlers={{
                        click: () =>
                          dialogHandler({
                            title: marker.name ?? `Marker ${marker.id}`,
                            id: marker.id,
                            name: marker.name,
                            latitude: marker.latitude,
                            longitude: marker.longitude,
                            createdAt: marker.createdAt,
                            updatedAt: marker.updatedAt,
                          }),
                      }}
                    ></Marker>
                  ))}
                </>
              </LayersControl.Overlay>
            </LayersControl>
            <MapClickHandler onMapClick={dialogHandler} />{" "}
          </MapContainer>
        );
      };
    }),
  {
    ssr: false,
  },
);

export default function InteractiveMap() {
  const [selectedData, setSelectedData] = useState<DialogData | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isClient, setIsClient] = useState(false);

  // Fetch markers from the database
  const { data: markers = [], isLoading: markersLoading } =
    api.markers.getAll.useQuery();

  useEffect(() => {
    setIsClient(true);
  }, []);
  const showDialog = (data: DialogData) => {
    setSelectedData(data);
    setIsDialogOpen(true);
  };
  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error("Failed to copy: ", err);
    }
  };

  const formatValue = (
    value: string | number | boolean | Date | null | undefined,
  ): string => {
    if (value === null || value === undefined) {
      return "N/A";
    }
    if (value instanceof Date) {
      return value.toLocaleDateString();
    }
    if (typeof value === "number") {
      // Format numbers with up to 6 decimal places if they're coordinates
      return value.toFixed(6).replace(/\.?0+$/, "");
    }
    if (typeof value === "boolean") {
      return value ? "Yes" : "No";
    }
    return String(value);
  };
  if (!isClient || markersLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-gray-100">
        <div className="text-lg">Loading map...</div>
      </div>
    );
  }

  return (
    <>
      <div className="h-screen w-full">
        <MapContent dialogHandler={showDialog} markers={markers} />
      </div>{" "}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{selectedData?.title ?? "Data"}</DialogTitle>
          </DialogHeader>
          {selectedData && (
            <div className="space-y-4">
              {Object.entries(selectedData)
                .filter(([key]) => key !== "title") // Don't show title in the content
                .map(([key, value]) => (
                  <div key={key} className="space-y-2">
                    <div className="text-sm text-gray-600 capitalize">
                      {key.replace(/([A-Z])/g, " $1").trim()}:
                    </div>
                    <div className="rounded bg-gray-100 p-2 font-mono text-sm">
                      {formatValue(value)}
                    </div>
                  </div>
                ))}

              {/* Special handling for coordinates if they exist */}
              {selectedData.latitude && selectedData.longitude && (
                <>
                  <div className="space-y-2">
                    <div className="text-sm text-gray-600">
                      Formatted Coordinates:
                    </div>
                    <div className="rounded bg-gray-100 p-2 font-mono text-sm">
                      {formatValue(selectedData.latitude)},{" "}
                      {formatValue(selectedData.longitude)}
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <Button
                      onClick={() =>
                        copyToClipboard(
                          `${formatValue(selectedData.latitude)}, ${formatValue(selectedData.longitude)}`,
                        )
                      }
                      variant="outline"
                      size="sm"
                      className="flex-1"
                    >
                      Copy Coordinates
                    </Button>{" "}
                    <Button
                      onClick={() =>
                        copyToClipboard(
                          `https://www.openstreetmap.org/#map=18/${Number(selectedData.latitude)}/${Number(selectedData.longitude)}`,
                        )
                      }
                      variant="outline"
                      size="sm"
                      className="flex-1"
                    >
                      Copy OSM Link
                    </Button>
                  </div>
                </>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
