"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "~/components/ui/dialog";
import { Button } from "~/components/ui/button";

interface Coordinates {
  lat: number;
  lng: number;
}

// Create a separate component for the map content that will be dynamically loaded
const MapContent = dynamic(
  () =>
    import("react-leaflet").then((mod) => {
      const { MapContainer, TileLayer, useMapEvents } = mod;
      
      function MapClickHandler({ onMapClick }: { onMapClick: (coords: Coordinates) => void }) {
        useMapEvents({
          click: (e) => {
            onMapClick({ lat: e.latlng.lat, lng: e.latlng.lng });
          },
        });
        return null;
      }

      return function Map({ onMapClick }: { onMapClick: (coords: Coordinates) => void }) {
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
            <MapClickHandler onMapClick={onMapClick} />
          </MapContainer>
        );
      };
    }),
  {
    ssr: false,
  }
);

export default function InteractiveMap() {
  const [selectedCoords, setSelectedCoords] = useState<Coordinates | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  const handleMapClick = (coords: Coordinates) => {
    setSelectedCoords(coords);
    setIsDialogOpen(true);
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error("Failed to copy: ", err);
    }
  };

  if (!isClient) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-gray-100">
        <div className="text-lg">Loading map...</div>
      </div>
    );
  }

  return (
    <>
      <div className="h-screen w-full">
        <MapContent onMapClick={handleMapClick} />
      </div>

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Coordinates</DialogTitle>
          </DialogHeader>
          {selectedCoords && (
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="text-sm text-gray-600">Latitude:</div>
                <div className="rounded bg-gray-100 p-2 font-mono text-sm">
                  {selectedCoords.lat.toFixed(6)}
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-sm text-gray-600">Longitude:</div>
                <div className="rounded bg-gray-100 p-2 font-mono text-sm">
                  {selectedCoords.lng.toFixed(6)}
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-sm text-gray-600">Formatted:</div>
                <div className="rounded bg-gray-100 p-2 font-mono text-sm">
                  {selectedCoords.lat.toFixed(6)}, {selectedCoords.lng.toFixed(6)}
                </div>
              </div>
              <div className="flex space-x-2">
                <Button
                  onClick={() =>
                    copyToClipboard(`${selectedCoords.lat.toFixed(6)}, ${selectedCoords.lng.toFixed(6)}`)
                  }
                  variant="outline"
                  size="sm"
                  className="flex-1"
                >
                  Copy Coordinates
                </Button>
                <Button
                  onClick={() =>
                    copyToClipboard(
                      `https://www.openstreetmap.org/#map=18/${selectedCoords.lat}/${selectedCoords.lng}`
                    )
                  }
                  variant="outline"
                  size="sm"
                  className="flex-1"
                >
                  Copy OSM Link
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
