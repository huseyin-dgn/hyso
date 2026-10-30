"""P30 descriptive measurements from validated host events, never sensor FPS."""


def _latency(values, reason='no_completed_observations'):
    values=sorted(values)
    def percentile(q):
        position=(len(values)-1)*q;lower=int(position);upper=min(lower+1,len(values)-1)
        return values[lower]+(values[upper]-values[lower])*(position-lower)
    return dict(count=len(values),unit='ms',status='measured' if values else 'unsupported',
        reason=None if values else reason,mean=sum(values)/len(values) if values else None,
        p50=percentile(.5) if values else None,p95=percentile(.95) if values else None,
        percentile_method='linear_interpolation_at_(n-1)*q')


def _diagnostics(e, analysis):
    rows=e['rows'];header=e['header'] or dict(clock_id=None,capture=dict(declared_fps=None))
    received={};enqueued={}
    samples={name:[] for name in ('capture_read','queue_wait','preprocess','inference','postprocess',
                                'processing_service','receive_to_completion')}
    read_starts=[];read_ends=[];inference_ns=0;inference_count=0
    for row in rows:
        frame=row.get('frame_id');stamp=row['timestamp_ns'];kind=row['event']
        if kind=='received':
            received[frame]=stamp;read_starts.append(row['read_start_ns']);read_ends.append(row['read_end_ns'])
            samples['capture_read'].append((row['read_end_ns']-row['read_start_ns'])/1e6)
        elif kind=='enqueued':enqueued[frame]=stamp
        elif kind=='dequeued':
            samples['queue_wait'].append((stamp-enqueued[frame])/1e6)
        elif kind=='processed':
            samples['processing_service'].append((row['process_end_ns']-row['process_start_ns'])/1e6)
            samples['receive_to_completion'].append((stamp-received[frame])/1e6)
            for stage in row['stages']:
                duration=stage['end_ns']-stage['start_ns']
                if stage['name'] in ('preprocess','inference','postprocess'):
                    samples[stage['name']].append(duration/1e6)
                if stage['name']=='inference':inference_ns+=duration;inference_count+=1
    start=min(read_starts) if read_starts else None
    capture_seconds=(max(read_ends)-start)/1e9 if read_ends else None
    # Occupancy integrates the actual application FIFO over the full observed window.
    depth=0;previous=start;area=0
    for row in rows:
        if previous is not None:area+=depth*(row['timestamp_ns']-previous)
        previous=row['timestamp_ns'];depth=row['queue_depth']
    window=(rows[-1]['timestamp_ns']-start) if rows and start is not None else 0
    count=analysis['received'];processed=analysis['processed']
    latency={name:_latency(values,'postprocessing_not_instrumented' if name=='postprocess' else 'no_completed_observations')
             for name,values in samples.items()}
    return dict(capture_read_completion_rate_per_second=count/capture_seconds if capture_seconds and capture_seconds>0 else None,
        capture_rate_reason=None if capture_seconds and capture_seconds>0 else 'no_positive_committed_capture_window',
        capture_observation_seconds=capture_seconds,
        capture_rate_scope='successful_OpenCV_read_completions_over_first_read_start_to_last_read_end_includes_producer_stalls',
        completed_processing_fps=analysis['processed_per_second'],
        processing_rate_reason=None if analysis['processed_per_second'] is not None else 'full_run_completion_not_verified',
        processing_rate_scope='completed_frames_over_first_read_start_to_last_lifecycle_event_full_run_only',
        inference_calls_per_active_second=inference_count*1e9/inference_ns if inference_ns else None,
        inference_rate_reason=None if inference_ns else 'no_completed_positive_inference_brackets',
        inference_rate_scope='completed_forward_calls_over_summed_host_inference_brackets_excludes_capture_queue_and_other_stages',
        latency_ms=latency,drop_rate=analysis['dropped']/count if count else None,
        late_rate=analysis['late_processed']/processed if processed else None,
        late_rate_denominator='processed',late_rule='receive_to_completion_strictly_greater_than_late_after_seconds',
        drop_rate_denominator='received',drop_scope='application_received_frames_only',
        queue_occupancy=dict(time_weighted_mean_depth=area/window if window else None,final_depth=depth,
            capacity=e['request']['protocol']['queue_capacity'],scope='application_FIFO_excludes_inflight_and_producer_frame'),
        display_fps=None,display_fps_status='unsupported',display_fps_reason='no_display_presentation_events_measured',
        sensor_fps=None,sensor_fps_reason='host_read_completion_is_not_sensor_exposure',
        upstream_dropped_frames=None,upstream_drop_reason='driver_transport_and_sensor_losses_not_observable',
        video_metadata_fps=header['capture']['declared_fps'],
        measurement_clock=dict(clock_id=header['clock_id'],unit='ns',source='time.perf_counter_ns',
            status='measured' if rows else 'unavailable',reason=None if rows else 'no_committed_lifecycle_events',
            scope='worker_host_capture_queue_processing_events',media_pts_scope='separate_media_clock_ms_no_cross_clock_subtraction'),
        measurement_scope='full_run' if analysis['terminal_accounting_complete'] else 'committed_event_prefix_only_tail_unknown')
