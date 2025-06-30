import L from "leaflet";

export interface MarkerType {
  icon: L.DivIcon;
  description?: string;
}

const iconSize = 32; // Size of the icon in pixels

const schoolIcon = L.divIcon({
    html: `<svg xmlns="http://www.w3.org/2000/svg" width="${iconSize}" height="${iconSize}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-school-icon lucide-school"><path d="M14 22v-4a2 2 0 1 0-4 0v4"/><path d="m18 10 3.447 1.724a1 1 0 0 1 .553.894V20a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-7.382a1 1 0 0 1 .553-.894L6 10"/><path d="M18 5v17"/><path d="m4 6 7.106-3.553a2 2 0 0 1 1.788 0L20 6"/><path d="M6 5v17"/><circle cx="12" cy="9" r="2"/></svg>`,
    iconAnchor: [iconSize / 2, iconSize], // Center the icon at the bottom
    className: 'marker-icon school-icon'
});
const cameraIcon = L.divIcon({
    html: `<svg xmlns="http://www.w3.org/2000/svg" width="${iconSize}" height="${iconSize}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-video-icon lucide-video"><path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5"/><rect x="2" y="6" width="14" height="12" rx="2"/></svg>`,
    iconAnchor: [iconSize / 2, iconSize], // Center the icon at the bottom
    className: 'marker-icon camera-icon'
});
const defaultIcon = L.divIcon({
    html: `<svg xmlns="http://www.w3.org/2000/svg" width="${iconSize}" height="${iconSize}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-map-pin"><path d="M12 22s8-4.5 8-11A8 8 0 1 0 4 11c0 6.5 8 11 8 11Z"/><circle cx="12" cy="11" r="3"/></svg>`,
    iconAnchor: [iconSize / 2, iconSize], // Center the icon at the bottom
    className: 'marker-icon default-icon'
});

export const MARKER_TYPES: Record<string, MarkerType> = {
    PRIVATE_SCHOOL: {
        icon: schoolIcon,
        description: "Private educational institutions"
    },
    POST_SECONDARY_INSTITUTION: {
        icon: schoolIcon,
        description: "Universities, colleges, and other post-secondary institutions"
    },
    ENGLISH_PUBLIC_SCHOOL: {
        icon: schoolIcon,
        description: "English public schools"
    },
    ENGLISH_SEPARATE_SCHOOL: {
        icon: schoolIcon,
        description: "English separate schools"
    },
    FRENCH_PUBLIC_SCHOOL: {
        icon: schoolIcon,
        description: "French public schools"
    },
    FRENCH_SEPARATE_SCHOOL: {
        icon: schoolIcon,
        description: "French separate schools"
    },
    EXISTING_CAMERA: {
        icon: cameraIcon,
        description: "Traffic cameras and surveillance cameras"
    },
    // DEFAULT: {
    //     icon: defaultIcon,
    //     description: "Default marker type for general use"
    // },
        
} as const;

export type MarkerTypeId = keyof typeof MARKER_TYPES;

// Helper function to get marker type by id
export function getMarkerType(id: MarkerTypeId): MarkerType {
  return MARKER_TYPES[id]!;
}

// Helper function to get all marker types as an array
export function getAllMarkerTypes(): MarkerType[] {
  return Object.values(MARKER_TYPES);
}

// Helper function to get marker type names for dropdowns/selects
export function getMarkerTypeOptions(): Array<{ value: MarkerTypeId; label: string }> {
  return Object.entries(MARKER_TYPES).map(([key, type]) => ({
    value: key,
    // title case conversion for label
    label: key.toLowerCase().replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
  }));
}