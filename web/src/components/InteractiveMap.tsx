"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Button } from "~/components/ui/button";
import { RiskCategoryFilter } from "~/components/ui/risk-category-filter";
import { api } from "~/trpc/react";
import { Circle, LayerGroup, LayersControl } from "react-leaflet";
import { useMap } from "react-leaflet";
import type { Marker, Score } from "~/server/db/schema";

import type { LatLngTuple, LatLngBounds } from "leaflet";
import {
  getMarkerType,
  getMarkerTypeOptions,
  type MarkerTypeId,
} from "~/lib/markerType";
import {
  RISK_CATEGORIES,
  getRiskCategoryColor,
  shouldShowRiskCategory,
  type RiskCategory,
} from "~/lib/safetyScores";
const center: LatLngTuple = [43.6532, -79.3832]; // Default center for the map (Toronto)
const zoom = 13; // Default zoom level

interface DialogData
  extends Record<string, string | number | boolean | Date | null | undefined> {
  title?: string;
}

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
      const { MapContainer, TileLayer, useMapEvents, Marker, CircleMarker } = mod;
      
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

      function MapBoundsHandler({
        onBoundsChange,
      }: {
        onBoundsChange: (bounds: LatLngBounds) => void;
      }) {
        const map = useMap();
        
        useMapEvents({
          moveend: () => {
            onBoundsChange(map.getBounds());
          },
          zoomend: () => {
            onBoundsChange(map.getBounds());
          },
        });
        return null;
      }

      function renderSafetyScores(
        scores: Score[],
        selectedCategories: RiskCategory[],
        dialogHandler: (data: DialogData) => void,
      ) {
        return scores
          .filter(score => 
            score.risk_category && 
            shouldShowRiskCategory(score.risk_category, selectedCategories)
          )
          .map((score, index) => (
            <CircleMarker
              key={`${score.longitude}-${score.latitude}-${index}`}
              center={[score.latitude, score.longitude]}
              radius={6}
              pathOptions={{
                color: getRiskCategoryColor(score.risk_category ?? "Medium"),
                fillColor: getRiskCategoryColor(score.risk_category ?? "Medium"),
                fillOpacity: 0.7,
                weight: 2,
              }}
              eventHandlers={{
                click: () =>
                  dialogHandler({
                    title: `Safety Score - ${score.risk_category}`,
                    latitude: score.latitude,
                    longitude: score.longitude,
                    "Risk Category": score.risk_category,
                    "Final Score": score.final_score,
                    "Predicted Risk": score.predicted_risk,
                    "Collision Count": score.collision_count,
                    "Speed Risk": score.speed_risk,
                    "Volume Risk": score.volume_risk,
                    "Near School": score.near_school,
                    "In School Zone": score.in_school_zone,
                    "Has Camera": score.has_camera,
                    "Average Speed": score.avg_speed,
                    "85th Percentile Speed": score.avg_85th_percentile_speed,
                  }),
              }}
            />
          ));
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
      }      return function Map({
        dialogHandler,
        resetTrigger,
        selectedRiskCategories,
        safetyScores,
        onBoundsChange,
      }: {
        dialogHandler: (data: DialogData) => void;
        resetTrigger: number;
        selectedRiskCategories: RiskCategory[];
        safetyScores: Score[];
        onBoundsChange: (bounds: LatLngBounds) => void;
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

              <LayersControl.Overlay checked name="Safety Scores">
                <LayerGroup>
                  {renderSafetyScores(safetyScores, selectedRiskCategories, dialogHandler)}
                </LayerGroup>
              </LayersControl.Overlay>

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
            <MapClickHandler onMapClick={dialogHandler} />
            <MapBoundsHandler onBoundsChange={onBoundsChange} />
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
  const [resetTrigger, setResetTrigger] = useState(0);
  const [selectedRiskCategories, setSelectedRiskCategories] = useState<RiskCategory[]>([...RISK_CATEGORIES]);
  const [safetyScores, setSafetyScores] = useState<Score[]>([]);
  const [currentBounds, setCurrentBounds] = useState<LatLngBounds | null>(null);
  const [page, setPage] = useState(1);
  const [allScores, setAllScores] = useState<Score[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [hasMore, setHasMore] = useState(true);

  const paginatedQuery = api.scores.getScores.useQuery(
    {
      bounds: currentBounds ? {
        north: currentBounds.getNorth(),
        south: currentBounds.getSouth(),
        east: currentBounds.getEast(),
        west: currentBounds.getWest(),
      } : undefined,
      riskCategories: selectedRiskCategories,
      limit: 500, // Fetch 500 at a time
      offset: (page - 1) * 500,
    },
    {
      enabled: !!currentBounds, // Only run when bounds are available
      staleTime: 1000 * 60 * 5, // 5 minutes
      keepPreviousData: true,
    }
  );

  useEffect(() => {
    if (paginatedQuery.data) {
      const newData = paginatedQuery.data.data;
      // Append new, unique scores to the list
      setAllScores(prevScores => {
        const existingIds = new Set(prevScores.map(s => `${s.latitude}-${s.longitude}`));
        const uniqueNewScores = newData.filter(s => !existingIds.has(`${s.latitude}-${s.longitude}`));
        return [...prevScores, ...uniqueNewScores];
      });
      setTotalCount(paginatedQuery.data.pagination.total);
      setHasMore(paginatedQuery.data.pagination.hasMore);
    }
  }, [paginatedQuery.data]);

  const handleLoadMore = () => {
    if (hasMore && !paginatedQuery.isFetching) {
      setPage(prevPage => prevPage + 1);
    }
  };

  const handleBoundsChange = useCallback((bounds: LatLngBounds) => {
    setCurrentBounds(bounds);
    // Reset pagination when bounds change but keep existing scores
    setPage(1);
  }, []);

  const showDialog = (data: DialogData) => {
    setSelectedData(data);
    setIsDialogOpen(true);
  };

  const handleRiskCategoryChange = (categories: RiskCategory[]) => {
    setSelectedRiskCategories(categories);
    // Reset pagination when filters change
    setPage(1);
    setAllScores([]);
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
    if (typeof value === "boolean") {
      return value ? "Yes" : "No";
    }
    return String(value);
  };

  return (
    <div className="relative h-screen w-full">
      <MapContent
        dialogHandler={showDialog}
        resetTrigger={resetTrigger}
        selectedRiskCategories={selectedRiskCategories}
        safetyScores={allScores} // Use allScores for rendering
        onBoundsChange={handleBoundsChange}
      />
      <div className="absolute bottom-4 left-4 z-10 rounded-lg bg-white bg-opacity-80 p-4 shadow-lg">
        <h3 className="text-lg font-bold">Safety Score Filters</h3>
        <RiskCategoryFilter
          selectedCategories={selectedRiskCategories}
          onChange={handleRiskCategoryChange}
        />
        <div className="mt-4">
          <p>Showing <strong>{allScores.length}</strong> of <strong>{totalCount}</strong> points</p>
          {hasMore && (
            <Button 
              onClick={handleLoadMore} 
              disabled={paginatedQuery.isFetching}
              className="mt-2 w-full"
            >
              {paginatedQuery.isFetching ? "Loading..." : "Load More"}
            </Button>
          )}
        </div>
      </div>
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-h-[80vh] overflow-y-auto sm:max-w-md">
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
    </div>
  );
}
