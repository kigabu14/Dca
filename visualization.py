# dca/visualization.py
import plotly.graph_objects as go

def plot_radar_chart(score_data, title="คะแนนวิเคราะห์ DCA (ตามเกณฑ์บัฟเฟตต์)"):
    """
    แสดง Radar Chart จากข้อมูลคะแนนแต่ละเกณฑ์
    :param score_data: dict จาก analyze_dca()
    :return: plotly.graph_objects.Figure
    """
    labels = [c['name'] for c in score_data['criteria']]
    values = [c['score'] for c in score_data['criteria']]

    # ปิดกราฟเป็นวงกลม
    labels += [labels[0]]
    values += [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='คะแนน DCA',
        line=dict(color='orange')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=False,
        title=title
    )

    return fig


def plot_pie_chart(score_data):
    """
    แสดง Pie Chart รวมคะแนนแต่ละเกณฑ์
    :param score_data: dict จาก analyze_dca()
    :return: plotly.graph_objects.Figure
    """
    labels = [c['name'] for c in score_data['criteria']]
    values = [c['score'] for c in score_data['criteria']]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

    fig.update_traces(textinfo='label+percent', marker=dict(line=dict(color='#000000', width=1)))
    fig.update_layout(title='สัดส่วนคะแนนรวมแต่ละเกณฑ์')

    return fig