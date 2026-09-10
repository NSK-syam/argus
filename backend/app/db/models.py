"""SQLAlchemy models matching the entities in the build plan:
Dataset, PipelineRun, Attempt, ModelVersion, Deployment.

Each field maps directly onto something the orchestrator/run service
already produces (LoggedAttempt, TrustGateResult.as_evidence(), etc.) --
this is persistence, not a new data model invented from scratch.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)  # "bundled_fd001" | "upload"
    schema_json: Mapped[dict] = mapped_column(JSON, default=dict)
    profile_json: Mapped[dict] = mapped_column(JSON, default=dict)
    storage_path: Mapped[str] = mapped_column(String)
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    runs: Mapped[list["PipelineRun"]] = relationship(back_populates="dataset")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    goal: Mapped[str] = mapped_column(Text, default="")
    task_type: Mapped[str] = mapped_column(String, default="regression")
    status: Mapped[str] = mapped_column(String, default="pending")
    # pending | running | succeeded | failed
    winning_attempt_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stopped_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    dataset: Mapped["Dataset"] = relationship(back_populates="runs")
    attempts: Mapped[list["Attempt"]] = relationship(
        back_populates="run", order_by="Attempt.attempt_number", cascade="all, delete-orphan"
    )
    model_versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id"))
    attempt_number: Mapped[int] = mapped_column(Integer)
    plan_json: Mapped[dict] = mapped_column(JSON, default=dict)
    model_family: Mapped[str] = mapped_column(String)
    hyperparams_json: Mapped[dict] = mapped_column(JSON, default=dict)
    validation_metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    test_metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    conformal_coverage: Mapped[float] = mapped_column(Float, default=0.0)
    gate_json: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_passed: Mapped[bool] = mapped_column(default=False)
    revision_action: Mapped[str] = mapped_column(String, default="")
    revision_rationale: Mapped[str] = mapped_column(Text, default="")
    plan_source: Mapped[str] = mapped_column(String, default="deterministic_fallback")
    fallback_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped["PipelineRun"] = relationship(back_populates="attempts")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id"))
    attempt_number: Mapped[int] = mapped_column(Integer)
    model_family: Mapped[str] = mapped_column(String, default="")
    artifact_path: Mapped[str] = mapped_column(String)
    feature_columns_json: Mapped[list] = mapped_column(JSON, default=list)
    conformal_q: Mapped[float] = mapped_column(Float, default=0.0)
    shap_summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    trust_gate_passed: Mapped[bool] = mapped_column(default=False)
    stage: Mapped[str] = mapped_column(String, default="shadow")  # shadow | production
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped["PipelineRun"] = relationship(back_populates="model_versions")
    deployments: Mapped[list["Deployment"]] = relationship(
        back_populates="model_version", cascade="all, delete-orphan"
    )


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    stage: Mapped[str] = mapped_column(String)  # shadow | production
    promoted_by: Mapped[str] = mapped_column(String, default="human_confirmed")
    audit_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    model_version: Mapped["ModelVersion"] = relationship(back_populates="deployments")
