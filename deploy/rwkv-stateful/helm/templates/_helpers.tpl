{{- define "rwkv-stateful.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "rwkv-stateful.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "rwkv-stateful.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "rwkv-stateful.labels" -}}
app.kubernetes.io/name: {{ include "rwkv-stateful.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
