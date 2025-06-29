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
import L, { marker } from "leaflet";
import { Circle, LayerGroup, LayersControl } from "react-leaflet";
import { useMap } from "react-leaflet";
import type { Marker } from "~/server/db/schema"; // Adjust the import path based on your project structure
// Adjust the import path based on your project structure

import type { LatLngTuple } from "leaflet";
import {
  getMarkerType,
  getMarkerTypeOptions,
  MARKER_TYPES,
  type MarkerTypeId,
} from "~/lib/markerType";
const center: LatLngTuple = [43.6532, -79.3832]; // Default center for the map (Toronto)
const zoom = 13; // Default zoom level

interface DialogData
  extends Record<string, string | number | boolean | Date | null | undefined> {
  title?: string;
}

const iconSize = 32; // Size of the icon in pixels

function ResetMapView({
  center,
  zoom,
  trigger,
}: {
  center: LatLngTuple;
  zoom: number;
  trigger: number;
}) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [trigger, center, zoom, map]);
  return null;
}

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

      function renderMarkerCategory(
        markerOption: {
          value: MarkerTypeId;
          label: string;
        },
        dialogHandler: (data: DialogData) => void,
      ) {
        const markers = api.markers.getByType.useQuery({
          markerType: markerOption.value,
        });
        if (markers.isLoading) {
          console.log(`Loading markers for type: ${markerOption.label}`);
          return null; // Handle loading state if needed
        }
        if (markers.isError) {
          console.error(`Error loading markers for type: ${markerOption.label}`, markers.error);
          return <div>Error loading markers</div>;
        }
        if (!markers.data || markers.data.length === 0) {
          console.log(`No markers found for type: ${markerOption.label}`);
          return null; // No markers to display
        }
        console.log(`Loaded ${markers.data.length} markers for type: ${markerOption.label}`, markers.data);

        return markers.data.map((marker: Marker) => (
          <Marker
            key={marker.id}
            position={[marker.latitude, marker.longitude]}
            icon={getMarkerType(markerOption.value).icon}
            eventHandlers={{
              click: () =>
                dialogHandler({
                  title: marker.name ?? `Marker ${marker.id}`,
                  id: marker.id,
                  name: marker.name,
                  latitude: marker.latitude,
                  longitude: marker.longitude,
                  markerType: marker.markerType,
                }),
            }}
          ></Marker>
        ));
      }

      return function Map({
        dialogHandler,
        resetTrigger,
      }: {
        dialogHandler: (data: DialogData) => void;
        resetTrigger: number;
      }) {
        return (
          <MapContainer
            center={center} // Default to toronto
            zoom={zoom}
            style={{ height: "100%", width: "100%" }}
            className="z-0"
          >
            <ResetMapView center={center} zoom={zoom} trigger={resetTrigger} />
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
              url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
            />
            <LayersControl position="topright">
              {getMarkerTypeOptions().map((MarkerOption) => (
                <LayersControl.Overlay
                  name={MarkerOption.label}
                  key={MarkerOption.value}
                >
                  <LayerGroup>
                    {renderMarkerCategory(MarkerOption, dialogHandler)}
                  </LayerGroup>
                </LayersControl.Overlay>
              ))}

              <LayersControl.Overlay checked name="Existing Cameras">
                {/* PLACEHOLDER */}
                <Circle
                  center={center}
                  radius={300}
                  pathOptions={{
                    color: "red",
                    fillColor: "red",
                    fillOpacity: 0.1,
                  }}
                />
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
  const [resetTrigger, setResetTrigger] = useState(0);

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

  return (
    <>
      <div className="h-screen w-full">
        <Button
          onClick={() => setResetTrigger((t) => t + 1)}
          variant="outline"
          size="sm"
          className="absolute right-4 bottom-4 z-10"
        >
          Reset View
        </Button>
        <MapContent dialogHandler={showDialog} resetTrigger={resetTrigger} />
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
