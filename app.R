# =============================================================================
# MRI-PDFF Prediction Calculator
# Separately predicts continuous MRI-PDFF (%) and steatosis grade (S0–S3) 
# using Random Forest models
# 
# Developed by: HS Zhang, DY Kim, BK Kim
# Dept. of Biomedical Systems Informatics, Dept. of Hepatology
# Yonsei University College of Medicine, Seoul, South Korea
# =============================================================================

library(shiny)
library(bslib)
library(ranger)
library(caret)
library(ggplot2)

# --- Load saved models ---
rf_continuous   <- readRDS("rf_continuous_model.rds")
rf_multicat     <- readRDS("rf_multicat_model.rds")

# --- Predictor ranges for input validation ---
predictor_info <- list(
  CAP       = list(label = "CAP (dB/m)",           min = 100,  max = 400,  step = 1,    default = 250,  unit = "dB/m"),
  age       = list(label = "Age (years)",           min = 5,   max = 95,   step = 1,    default = 50,   unit = "years"),
  sex       = list(label = "Sex",                   choices = c("Male" = 1, "Female" = 0)),
  DM        = list(label = "Diabetes Mellitus",     choices = c("No" = 0, "Yes" = 1)),
  HTN       = list(label = "Hypertension",          choices = c("No" = 0, "Yes" = 1)),
  BMI       = list(label = "BMI (kg/m²)",           min = 15,   max = 60,   step = 0.1,  default = 25.0, unit = "kg/m²"),
  LSM       = list(label = "LSM (kPa)",             min = 2.0,  max = 75.0, step = 0.1,  default = 5.5,  unit = "kPa"),
  fastGlu   = list(label = "Fasting Glucose (mg/dL)", min = 50, max = 500,  step = 1,    default = 100,  unit = "mg/dL"),
  totChol   = list(label = "Total Cholesterol (mg/dL)", min = 50, max = 400, step = 1,   default = 200,  unit = "mg/dL"),
  HDL       = list(label = "HDL (mg/dL)",           min = 10,   max = 120,  step = 1,    default = 50,   unit = "mg/dL"),
  plt       = list(label = "Platelets (×10³/µL)",   min = 30,   max = 600,  step = 1,    default = 220,  unit = "×10³/µL"),
  albumin   = list(label = "Albumin (g/dL)",        min = 1.5,  max = 6.0,  step = 0.1,  default = 4.2,  unit = "g/dL"),
  AST       = list(label = "AST (U/L)",             min = 5,    max = 500,  step = 1,    default = 25,   unit = "U/L"),
  ALT       = list(label = "ALT (U/L)",             min = 5,    max = 500,  step = 1,    default = 25,   unit = "U/L"),
  totBili   = list(label = "Total Bilirubin (mg/dL)", min = 0.1, max = 30.0, step = 0.1, default = 0.8,  unit = "mg/dL")
)

# --- Steatosis grade definitions ---
grade_info <- data.frame(
  grade   = c("S0", "S1", "S2", "S3"),
  label   = c("No steatosis", "Mild steatosis", "Moderate steatosis", "Severe steatosis"),
  color   = c("#2E86AB", "#A0C878", "#F0A500", "#D7263D"),
  pdff_range = c("<5%", "5–16%", "16–22%", "≥22%"),
  stringsAsFactors = FALSE
)

# --- CSS ---
custom_css <- "
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700&family=DM+Serif+Display&display=swap');

:root {
  --clr-s0: #2E86AB;
  --clr-s1: #A0C878;
  --clr-s2: #F0A500;
  --clr-s3: #D7263D;
  --clr-bg: #F7F8FA;
  --clr-card: #FFFFFF;
  --clr-accent: #1B4965;
  --clr-text: #2D3436;
  --clr-muted: #6C757D;
  --clr-border: #E9ECEF;
}

body {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--clr-bg);
  color: var(--clr-text);
}

.app-header {
  background: linear-gradient(135deg, #1B4965 0%, #2E86AB 100%);
  color: white;
  padding: 28px 32px 22px;
  margin: -1rem -1rem 24px -1rem;
  border-radius: 0 0 16px 16px;
}

.app-header h1 {
  font-family: 'DM Serif Display', serif;
  font-size: 1.85rem;
  margin-bottom: 4px;
  letter-spacing: -0.02em;
}

.app-header .subtitle {
  font-size: 0.92rem;
  opacity: 0.85;
  font-weight: 400;
}

.app-header .credit {
  font-size: 0.78rem;
  opacity: 0.65;
  margin-top: 8px;
  line-height: 1.4;
}

.input-section-title {
  font-weight: 600;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--clr-accent);
  border-bottom: 2px solid var(--clr-accent);
  padding-bottom: 6px;
  margin-bottom: 14px;
  margin-top: 20px;
}

.input-section-title:first-of-type {
  margin-top: 4px;
}

.result-card {
  background: var(--clr-card);
  border: 1px solid var(--clr-border);
  border-radius: 14px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.04);
  margin-bottom: 18px;
  transition: box-shadow 0.25s ease;
}

.result-card:hover {
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

.result-card h3 {
  font-family: 'DM Serif Display', serif;
  font-size: 1.15rem;
  color: var(--clr-accent);
  margin-bottom: 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--clr-border);
}

.big-number {
  font-size: 2.8rem;
  font-weight: 700;
  line-height: 1.1;
  margin-bottom: 4px;
}

.big-number-unit {
  font-size: 1.1rem;
  font-weight: 400;
  color: var(--clr-muted);
}

.grade-badge {
  display: inline-block;
  font-size: 1.6rem;
  font-weight: 700;
  padding: 10px 28px;
  border-radius: 10px;
  color: white;
  margin-bottom: 4px;
}

.grade-description {
  font-size: 0.95rem;
  color: var(--clr-muted);
  margin-top: 4px;
}

.pdff-bar-container {
  position: relative;
  width: 100%;
  height: 28px;
  background: linear-gradient(to right, 
    var(--clr-s0) 0%, var(--clr-s0) 12.5%, 
    var(--clr-s1) 12.5%, var(--clr-s1) 40%, 
    var(--clr-s2) 40%, var(--clr-s2) 55%, 
    var(--clr-s3) 55%, var(--clr-s3) 100%);
  border-radius: 6px;
  margin: 16px 0 6px;
  overflow: visible;
}

.pdff-marker {
  position: absolute;
  top: -6px;
  width: 4px;
  height: 40px;
  background: var(--clr-text);
  border-radius: 2px;
  transition: left 0.4s ease;
}

.pdff-marker::after {
  content: '▼';
  position: absolute;
  top: -18px;
  left: -6px;
  font-size: 14px;
  color: var(--clr-text);
}

.pdff-bar-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.72rem;
  color: var(--clr-muted);
  margin-top: 2px;
}

.prob-bar-wrap {
  margin-bottom: 10px;
}

.prob-bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  font-weight: 500;
  margin-bottom: 3px;
}

.prob-bar-track {
  width: 100%;
  height: 24px;
  background: #EDF2F7;
  border-radius: 6px;
  overflow: hidden;
}

.prob-bar-fill {
  height: 100%;
  border-radius: 6px;
  transition: width 0.5s ease;
  min-width: 2px;
}

.disclaimer-box {
  background: #FFF8E1;
  border: 1px solid #FFE082;
  border-radius: 10px;
  padding: 14px 18px;
  font-size: 0.82rem;
  color: #6D5B00;
  margin-top: 18px;
  line-height: 1.5;
}

.disclaimer-box strong {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85rem;
}

.validation-warning {
  color: #D7263D;
  font-size: 0.8rem;
  font-weight: 500;
  margin-top: 4px;
}

.predict-btn {
  width: 100%;
  padding: 12px;
  font-size: 1.05rem;
  font-weight: 600;
  border-radius: 10px;
  margin-top: 20px;
  background: linear-gradient(135deg, #1B4965, #2E86AB);
  border: none;
  color: white;
  cursor: pointer;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.predict-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(27,73,101,0.3);
}

.predict-btn:active {
  transform: translateY(0px);
}

.input-sidebar .form-group {
  margin-bottom: 12px;
}

.input-sidebar .form-control,
.input-sidebar .shiny-input-container select {
  border-radius: 8px;
  border: 1px solid var(--clr-border);
  font-size: 0.9rem;
}

.waiting-message {
  text-align: center;
  color: var(--clr-muted);
  font-size: 1rem;
  padding: 40px 20px;
  font-style: italic;
}
"

# =============================================================================
# UI
# =============================================================================
ui <- page_fluid(
  tags$head(tags$style(HTML(custom_css))),
  
  # Header
  div(class = "app-header",
    h1("MRI-PDFF Prediction Calculator"),
    div(class = "subtitle",
      "Random Forest-based prediction from clinical parameters"
    ),
    div(class = "credit",
      "DY Kim, HS Zhang, and BK Kim (Correspondence: BEOMKKIM@yuhs.ac)",
      tags$br(),
      "Yonsei University College of Medicine, Seoul, South Korea"
    )
  ),
  
  layout_columns(
    col_widths = c(4, 8),
    
    # --- Left: Input Panel ---
    div(class = "input-sidebar",
      div(class = "result-card",
        h3("Patient Parameters"),
        
        # Demographics
        div(class = "input-section-title", "Demographics"),
        layout_columns(
          col_widths = c(6, 6),
          numericInput("age", "Age (years)", value = 50, min = 5, max = 95, step = 1),
          selectInput("sex", "Sex", choices = c("Male" = 1, "Female" = 0))
        ),
        layout_columns(
          col_widths = c(6, 6),
          numericInput("BMI", "BMI (kg/m²)", value = 25.0, min = 15, max = 60, step = 0.1),
          NULL
        ),
        
        # Comorbidities
        div(class = "input-section-title", "Comorbidities"),
        layout_columns(
          col_widths = c(6, 6),
          selectInput("DM", "Diabetes Mellitus", choices = c("No" = 0, "Yes" = 1)),
          selectInput("HTN", "Hypertension", choices = c("No" = 0, "Yes" = 1))
        ),
        
        # FibroScan
        div(class = "input-section-title", "FibroScan"),
        layout_columns(
          col_widths = c(6, 6),
          numericInput("CAP", "CAP (dB/m)", value = 250, min = 100, max = 400, step = 1),
          numericInput("LSM", "LSM (kPa)", value = 5.5, min = 2.0, max = 75.0, step = 0.1)
        ),
        
        # Laboratory
        div(class = "input-section-title", "Laboratory"),
        layout_columns(
          col_widths = c(6, 6),
          numericInput("AST", "AST (U/L)", value = 25, min = 5, max = 500, step = 1),
          numericInput("ALT", "ALT (U/L)", value = 25, min = 5, max = 500, step = 1)
        ),
        layout_columns(
          col_widths = c(6, 6),
          numericInput("fastGlu", "Fasting Glucose (mg/dL)", value = 100, min = 50, max = 500, step = 1),
          numericInput("totChol", "Total Cholesterol (mg/dL)", value = 200, min = 50, max = 400, step = 1)
        ),
        layout_columns(
          col_widths = c(6, 6),
          numericInput("HDL", "HDL (mg/dL)", value = 50, min = 10, max = 120, step = 1),
          numericInput("plt", "Platelets (×10³/µL)", value = 220, min = 30, max = 600, step = 1)
        ),
        layout_columns(
          col_widths = c(6, 6),
          numericInput("albumin", "Albumin (g/dL)", value = 4.2, min = 1.5, max = 6.0, step = 0.1),
          numericInput("totBili", "Total Bilirubin (mg/dL)", value = 0.8, min = 0.1, max = 30.0, step = 0.1)
        ),
        
        # Predict button
        actionButton("predict_btn", "MRI-PDFF Prediction",
                     class = "predict-btn",
                     icon = icon("calculator")),
        
        # Validation warnings
        uiOutput("validation_warnings")
      )
    ),
    
    # --- Right: Results Panel ---
    div(
      # Waiting state
      uiOutput("results_ui")
    )
  )
)

# =============================================================================
# Server
# =============================================================================
server <- function(input, output, session) {
  
  # --- Input validation ---
  validate_inputs <- reactive({
    warnings <- c()
    
    checks <- list(
      list(id = "CAP",     val = input$CAP,     lo = 100,  hi = 400,  name = "CAP"),
      list(id = "age",     val = input$age,     lo = 5,   hi = 95,   name = "Age"),
      list(id = "BMI",     val = input$BMI,     lo = 15,   hi = 60,   name = "BMI"),
      list(id = "LSM",     val = input$LSM,     lo = 2,    hi = 75,   name = "LSM"),
      list(id = "fastGlu", val = input$fastGlu, lo = 50,   hi = 500,  name = "Fasting glucose"),
      list(id = "totChol", val = input$totChol, lo = 50,   hi = 400,  name = "Total cholesterol"),
      list(id = "HDL",     val = input$HDL,     lo = 10,   hi = 120,  name = "HDL"),
      list(id = "plt",     val = input$plt,     lo = 30,   hi = 600,  name = "Platelets"),
      list(id = "albumin", val = input$albumin, lo = 1.5,  hi = 6.0,  name = "Albumin"),
      list(id = "AST",     val = input$AST,     lo = 5,    hi = 500,  name = "AST"),
      list(id = "ALT",     val = input$ALT,     lo = 5,    hi = 500,  name = "ALT"),
      list(id = "totBili", val = input$totBili, lo = 0.1,  hi = 30,   name = "Total bilirubin")
    )
    
    any_null <- FALSE
    for (ch in checks) {
      if (is.null(ch$val) || is.na(ch$val)) {
        warnings <- c(warnings, paste0(ch$name, ": value is missing."))
        any_null <- TRUE
      } else if (ch$val < ch$lo || ch$val > ch$hi) {
        warnings <- c(warnings, paste0(ch$name, ": ", ch$val, 
                                        " is outside the expected range (", ch$lo, "–", ch$hi, ")."))
      }
    }
    
    list(warnings = warnings, any_null = any_null)
  })
  
  output$validation_warnings <- renderUI({
    vw <- validate_inputs()
    if (length(vw$warnings) == 0) return(NULL)
    
    div(class = "validation-warning", style = "margin-top: 12px;",
      icon("exclamation-triangle"),
      tags$ul(style = "margin: 4px 0 0 0; padding-left: 18px;",
        lapply(vw$warnings, function(w) tags$li(w))
      )
    )
  })
  
  # --- Build new data for prediction ---
  build_newdata <- reactive({
    ast_val <- input$AST
    alt_val <- input$ALT
    ast_alt_ratio <- ifelse(!is.null(alt_val) && !is.na(alt_val) && alt_val > 0,
                            ast_val / alt_val, NA)
    
    data.frame(
      CAP       = as.numeric(input$CAP),
      age       = as.numeric(input$age),
      sex       = as.numeric(input$sex),
      DM        = as.numeric(input$DM),
      HTN       = as.numeric(input$HTN),
      BMI       = as.numeric(input$BMI),
      LSM       = as.numeric(input$LSM),
      fastGlu   = as.numeric(input$fastGlu),
      totChol   = as.numeric(input$totChol),
      HDL       = as.numeric(input$HDL),
      plt       = as.numeric(input$plt),
      albumin   = as.numeric(input$albumin),
      AST       = as.numeric(input$AST),
      ALT       = as.numeric(input$ALT),
      ASTLTratio = ast_alt_ratio,
      totBili   = as.numeric(input$totBili)
    )
  })
  
  # --- Run predictions on button click ---
  predictions <- eventReactive(input$predict_btn, {
    vw <- validate_inputs()
    if (vw$any_null) return(NULL)
    
    newdata <- build_newdata()
    
    # ---- CONTINUOUS MRI-PDFF PREDICTION ----
    # Using 'ranger' itself so NOT a vec.
    pdff_pred <- predict(rf_continuous, newdata)$predictions
    
    # ---- MULTICATEGORY GRADE PREDICTION ----
    prob_matrix <- predict(rf_multicat, newdata)$predictions #, type = "prob"
    prob_matrix <- as.data.frame(prob_matrix)
    # name the factor levels
    colnames(prob_matrix) <- c("S0", "S1", "S2", "S3")
    pred_grade_idx <- which.max(prob_matrix[1, ])
    pred_grade <- c("S0", "S1", "S2", "S3")[pred_grade_idx]
    
    list(
      pdff_continuous = pdff_pred,
      grade_probs     = as.numeric(prob_matrix[1, ]),
      pred_grade      = pred_grade,
      pred_grade_idx  = pred_grade_idx
    )
  })
  
  # --- Render results ---
  output$results_ui <- renderUI({
    pred <- predictions()
    
    if (is.null(pred)) {
      return(
        div(
          div(class = "result-card",
            div(class = "waiting-message",
              icon("stethoscope", style = "font-size: 2rem; color: #BCC3CE; margin-bottom: 12px;"),
              tags$br(),
              "Enter patient parameters and click ", 
              tags$strong("Calculate Prediction"), 
              " to see results."
            )
          ),
          # Disclaimer as always visible
          div(class = "disclaimer-box",
            tags$strong("⚠ Research Use Only"),
            "This calculator is intended for research purposes only and has not been validated ",
            "for clinical decision-making. Predictions are based on Random Forest models trained ",
            "on a specific study cohort and may not generalize to all patient populations. ",
            "MRI-PDFF remains the reference standard for hepatic fat quantification. ",
          )
        )
      )
    }
    
    # Grade info
    gi <- grade_info[pred$pred_grade_idx, ]
    probs <- pred$grade_probs
    
    # Continuous PDFF value
    pdff_val <- round(pred$pdff_continuous, 1)
    
    # Determine color for continuous value
    pdff_color <- if (pdff_val < 5) "#2E86AB" 
                  else if (pdff_val < 16) "#A0C878" 
                  else if (pdff_val < 22) "#F0A500" 
                  else "#D7263D"
    
    # Marker position on gradient bar (scale 0–40% PDFF to 0–100% width)
    marker_pct <- min(max(pdff_val / 40 * 100, 0), 100)
    
    div(
      # --- Card 1: Continuous MRI-PDFF ---
      div(class = "result-card",
        h3(icon("chart-line"), " Predicted MRI-PDFF (Continuous)"),
        
        div(style = "text-align: center; margin: 10px 0 8px;",
          div(class = "big-number", style = paste0("color:", pdff_color, ";"),
            pdff_val,
            span(class = "big-number-unit", " %")
          ),
          div(style = "font-size: 0.9rem; color: var(--clr-muted);",
            if (pdff_val < 5) "Below steatosis threshold (< 5%)"
            else paste0("Above steatosis threshold (≥ 5%)")
          )
        ),
        
        # Gradient bar with marker
        div(class = "pdff-bar-container",
          div(class = "pdff-marker", 
              style = paste0("left: calc(", marker_pct, "% - 2px);"))
        ),
        div(class = "pdff-bar-labels",
          span("0%"), span("5%"), span("16.3%"), span("21.7%"), span("≥40%")
        ),
        div(style = "display: flex; justify-content: space-around; margin-top: 6px; font-size: 0.75rem;",
          span(style = "color: var(--clr-s0); font-weight: 600;", "S0"),
          span(style = "color: var(--clr-s1); font-weight: 600;", "S1"),
          span(style = "color: var(--clr-s2); font-weight: 600;", "S2"),
          span(style = "color: var(--clr-s3); font-weight: 600;", "S3")
        )
      ),
      
      # --- Card 2: Multicategory Grade ---
      div(class = "result-card",
        h3(icon("layer-group"), " Predicted Steatosis Grade (S0–S3)"),
        
        div(style = "text-align: center; margin: 10px 0 16px;",
          div(class = "grade-badge", 
              style = paste0("background-color:", gi$color, ";"),
              pred$pred_grade),
          div(class = "grade-description", gi$label),
          div(style = "font-size: 0.82rem; color: var(--clr-muted); margin-top: 2px;",
              paste0("MRI-PDFF range: ", gi$pdff_range))
        ),
        
        # Probability bars
        div(style = "margin-top: 16px;",
          div(style = "font-weight: 600; font-size: 0.88rem; margin-bottom: 10px; color: var(--clr-accent);",
              "Class Probabilities"),
          
          lapply(1:4, function(i) {
            gname <- grade_info$grade[i]
            glabel <- grade_info$label[i]
            gcolor <- grade_info$color[i]
            prob_pct <- round(probs[i] * 100, 1)
            is_max <- (i == pred$pred_grade_idx)
            
            div(class = "prob-bar-wrap",
              div(class = "prob-bar-label",
                span(style = if (is_max) "font-weight: 700;" else "",
                     paste0(gname, " — ", glabel)),
                span(style = if (is_max) "font-weight: 700;" else "",
                     paste0(prob_pct, "%"))
              ),
              div(class = "prob-bar-track",
                div(class = "prob-bar-fill",
                    style = paste0("width:", prob_pct, "%; background-color:", gcolor, ";",
                                   if (is_max) paste0(" box-shadow: 0 0 8px ", gcolor, "55;") else ""))
              )
            )
          })
        )
      ),
      
      # --- Disclaimer ---
      div(class = "disclaimer-box",
        tags$strong("⚠ Research Use Only"),
        "This calculator is intended for research purposes only and has not been externally validated ",
        "for clinical decision-making. Predictions are based on Random Forest models trained ",
        "on a specific study cohort and may not generalize to other patient populations. ",
        "MRI-PDFF remains the reference standard for hepatic fat quantification. "#,
      )
    )
  })
}

# =============================================================================
# Run the app
# =============================================================================
shinyApp(ui = ui, server = server)