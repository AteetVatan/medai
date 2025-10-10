/**
 * Lightweight React component blueprint for the Behandlungsbericht form.
 * Currently unused by the vanilla JS frontend but documented for future integration.
 */

import { ChangeEvent, useCallback } from "react";

export interface ReportFormValues {
  doctor_name: string;
  patient_name: string;
  patient_dob: string;
  prescription_date: string;
  treatment_date_from: string;
  treatment_date_to: string;
  physiotherapist_name: string;
  report_city: string;
  report_date: string;
  insurance_type: "PRIVAT" | "GESETZLICH" | "UNKLAR";
  diagnoses: string[];
  prescribed_therapy_type: string;
  patient_problem_statement: string;
  treatment_outcome: "BESCHWERDEFREI" | "LINDERUNG" | "KEINE_BESSERUNG" | "UNKLAR";
  therapy_status_note: string;
  follow_up_recommendation: string;
}

export interface ReportFormProps {
  values: ReportFormValues;
  onChange: (values: ReportFormValues) => void;
  onSuggest: () => void;
  onSave: () => void;
  onDownload: () => void;
}

/**
 * Render the report form using controlled inputs.
 */
export function ReportForm({ values, onChange, onSuggest, onSave, onDownload }: ReportFormProps) {
  const handleChange = useCallback(
    (event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
      const { name, value } = event.target;
      const next: ReportFormValues = {
        ...values,
        [name]: name === "diagnoses" ? value.split("\n").map((item) => item.trim()).filter(Boolean) : value,
      } as ReportFormValues;
      onChange(next);
    },
    [values, onChange]
  );

  return (
    <form className="report-form" onSubmit={(event) => event.preventDefault()}>
      <fieldset className="report-form__section" disabled>
        <legend>Vorgaben</legend>
        <label>
          Arzt
          <input type="text" name="doctor_name" value={values.doctor_name} readOnly />
        </label>
        <label>
          Patient
          <input type="text" name="patient_name" value={values.patient_name} readOnly />
        </label>
        <label>
          Geburtsdatum
          <input type="text" name="patient_dob" value={values.patient_dob} readOnly />
        </label>
        <label>
          Verordnung
          <input type="text" name="prescription_date" value={values.prescription_date} readOnly />
        </label>
        <label>
          Behandlungszeitraum
          <input type="text" name="treatment_range" value={`${values.treatment_date_from} bis ${values.treatment_date_to}`} readOnly />
        </label>
        <label>
          Therapeut
          <input type="text" name="physiotherapist_name" value={values.physiotherapist_name} readOnly />
        </label>
      </fieldset>

      <fieldset className="report-form__section">
        <legend>Bericht</legend>
        <label>
          Ort
          <input type="text" name="report_city" value={values.report_city} onChange={handleChange} />
        </label>
        <label>
          Datum
          <input type="date" name="report_date" value={values.report_date} onChange={handleChange} />
        </label>
        <label>
          Versicherung
          <select name="insurance_type" value={values.insurance_type} onChange={handleChange}>
            <option value="PRIVAT">Privat</option>
            <option value="GESETZLICH">Gesetzlich</option>
            <option value="UNKLAR">Unklar</option>
          </select>
        </label>
        <label>
          Diagnosen
          <textarea name="diagnoses" value={values.diagnoses.join("\n")} onChange={handleChange} rows={3} />
        </label>
        <label>
          Verordnete Therapie
          <input type="text" name="prescribed_therapy_type" value={values.prescribed_therapy_type} onChange={handleChange} />
        </label>
        <label>
          Problemformulierung
          <textarea name="patient_problem_statement" value={values.patient_problem_statement} onChange={handleChange} rows={3} />
        </label>
        <label>
          Ergebnis
          <select name="treatment_outcome" value={values.treatment_outcome} onChange={handleChange}>
            <option value="BESCHWERDEFREI">Beschwerdefrei</option>
            <option value="LINDERUNG">Linderung</option>
            <option value="KEINE_BESSERUNG">Keine Besserung</option>
            <option value="UNKLAR">Unklar</option>
          </select>
        </label>
        <label>
          Therapieverlauf
          <textarea name="therapy_status_note" value={values.therapy_status_note} onChange={handleChange} rows={3} />
        </label>
        <label>
          Empfehlung
          <textarea name="follow_up_recommendation" value={values.follow_up_recommendation} onChange={handleChange} rows={3} />
        </label>
      </fieldset>

      <div className="report-form__actions">
        <button type="button" onClick={onSuggest}>
          Aus Transkript ausfüllen
        </button>
        <button type="button" onClick={onSave}>
          Speichern
        </button>
        <button type="button" onClick={onDownload}>
          PDF herunterladen
        </button>
      </div>
    </form>
  );
}

export default ReportForm;
