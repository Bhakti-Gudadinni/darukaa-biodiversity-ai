from pydantic import BaseModel, Field
from typing import List, Optional

class ImpactedMetric(BaseModel):
    metric_name: str = Field(description="e.g., Soil Organic Carbon (SOC), Soil Moisture, Pollinator Abundance")
    quantified_impact: str = Field(description="e.g., +15-25% over 2-3 years")

class ScientificRecommendation(BaseModel):
    intervention: str = Field(description="Actionable practice, e.g., Legume cover cropping or Agroforestry shelterbelts")
    scientific_mechanism: str = Field(description="Explicit geochemical or biological explanation of why it works")
    impacted_metrics: List[ImpactedMetric]
    time_horizon: str = Field(description="Short-term (0-6 mo), Medium-term (6-24 mo), or Long-term (2-5 yr)")
    sources_cited: List[str] = Field(description="Authoritative sources, e.g., FAO Land & Water (2020), IPCC WGII (2022)")
    confidence_level: str = Field(description="High, Medium, or Low")

class SystemOutput(BaseModel):
    needs_clarification: bool = Field(description="True if user query misses baseline metrics like SOC %, rainfall, or land cover")
    clarification_message: Optional[str] = Field(default=None, description="The specific follow-up question to ask")
    recommendations: Optional[List[ScientificRecommendation]] = Field(default=None)