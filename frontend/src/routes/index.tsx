import { createFileRoute } from "@tanstack/react-router";
import Home from "@/pages/Home";

const title = "SatQuery AI — Remote Sensing Intelligence";
const description =
  "Interactive vision-language assistant for SAR analysis, change detection, and spatial grounding on satellite imagery.";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title },
      { name: "description", content: description },
      { property: "og:title", content: title },
      { property: "og:description", content: description },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Home,
});
